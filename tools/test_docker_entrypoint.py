"""docker-entrypoint.sh keeps the install config on the /data volume as the one that counts (#1244): with REINSTALL=0
and a config on the volume, /opt/strata/strata-<tag>.json is a link to it, even when a regular file is already there.
Runs the real script in a temporary tree with a stub for .venv/bin/python (skipped where there is no sh).

    python -m unittest tools.test_docker_entrypoint
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SH = shutil.which("sh")
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# The stub "python". setup.py --setup writes the config where setup would: strata-<tag>.json in the Strata folder,
# named by setup's own config_tag (through a link that is there). A plain start prints the config setup.py would start:
# the one strata-glm-*.json there, or that it would ask which one when there are several.
STUB = f"""#!{sys.executable}
import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
import setup
args, here = sys.argv[1:], Path.cwd()
if args[1:2] == ["--setup"]:
    given = args[args.index("--model") + 1]
    tag = setup.config_tag(setup.size_name(given), given, here)
    (here / f"strata-{{tag}}.json").write_text('{{"args": ["from-setup"]}}')
    print("SETUP: " + " ".join(args))
else:
    cfgs = sorted(setup.model_config_files(here))
    if len(cfgs) == 1:
        print("STARTED WITH: " + cfgs[0].read_text().strip() + " ARGS: " + " ".join(args))
    else:
        print("ASKS WHICH ONE: " + " ".join(p.name for p in cfgs))
"""


@unittest.skipIf(SH is None, "needs a POSIX sh")
class Entrypoint(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.opt = self.tmp / "opt"
        self.data = self.tmp / "data"
        (self.opt / ".venv" / "bin").mkdir(parents=True)
        (self.data / "config").mkdir(parents=True)
        stub = self.opt / ".venv" / "bin" / "python"
        stub.write_text(STUB, encoding="utf-8")
        stub.chmod(0o755)
        script = (ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8").replace("/opt/strata", str(self.opt))
        self.script = self.tmp / "entry.sh"
        self.script.write_text(script, encoding="utf-8")

    def run_entry(self, reinstall, **extra):
        env = dict(os.environ, STRATA_DATA=str(self.data), MODEL="UD-IQ3_XXS", REINSTALL=reinstall)
        env.update(extra)
        r = subprocess.run([SH, str(self.script)], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_edited_volume_config_wins_over_a_regular_file_in_opt(self):
        (self.data / "config" / "strata-glm-ud-iq3_xxs.json").write_text('{"args": ["edited", "--ctx", "262144"]}\n')
        (self.opt / "strata-glm-ud-iq3_xxs.json").write_text('{"args": ["stale"]}\n')      # left by an earlier setup
        out = self.run_entry("0")
        self.assertIn("edited", out)
        self.assertNotIn("stale", out)
        self.assertTrue((self.opt / "strata-glm-ud-iq3_xxs.json").is_symlink())

    def test_link_is_made_when_there_is_no_file(self):
        (self.data / "config" / "strata-glm-ud-iq3_xxs.json").write_text('{"args": ["on-volume"]}\n')
        self.assertIn("on-volume", self.run_entry("0"))

    def test_first_setup_copies_the_config_to_the_volume_and_the_next_start_links_it(self):
        out = self.run_entry("0")                                    # no config on the volume: setup runs
        self.assertIn("from-setup", out)
        cfg = self.data / "config" / "strata-glm-ud-iq3_xxs.json"
        self.assertIn("from-setup", cfg.read_text())
        cfg.write_text('{"args": ["edited-later"]}\n')
        self.assertIn("edited-later", self.run_entry("0"))           # the regular file setup left is replaced by the link

    def test_existing_link_into_the_data_config_dir_is_kept(self):      # a pod command that picked a config by linking
        (self.data / "config" / "strata-glm-ud-iq3_xxs.json").write_text('{"args": ["plain"]}\n')
        other = self.data / "config" / "strata-glm-ud-iq3_xxs.2.json"
        other.write_text('{"args": ["long-context"]}\n')
        os.symlink(other, self.opt / "strata-glm-ud-iq3_xxs.json")
        out = self.run_entry("0")
        self.assertIn("long-context", out)
        self.assertNotIn("plain", out)
        self.assertIn(f"Config: {other}", out)

    def test_link_to_somewhere_else_is_replaced(self):
        (self.data / "config" / "strata-glm-ud-iq3_xxs.json").write_text('{"args": ["on-volume"]}\n')
        elsewhere = self.tmp / "elsewhere.json"
        elsewhere.write_text('{"args": ["elsewhere"]}\n')
        os.symlink(elsewhere, self.opt / "strata-glm-ud-iq3_xxs.json")
        self.assertIn("on-volume", self.run_entry("0"))

    def test_config_env_wins(self):
        (self.data / "config" / "strata-glm-ud-iq3_xxs.json").write_text('{"args": ["plain"]}\n')
        other = self.tmp / "mine.json"
        other.write_text('{"args": ["mine"]}\n')
        os.symlink(self.data / "config" / "strata-glm-ud-iq3_xxs.json", self.opt / "strata-glm-ud-iq3_xxs.json")
        out = self.run_entry("0", CONFIG=str(other))
        self.assertIn("mine", out)
        self.assertNotIn("plain", out)
        self.assertIn(f"Config: {other}", out)

    def test_model_names_the_config_and_it_is_printed(self):
        (self.data / "config" / "strata-glm-ud-iq3_xxs.json").write_text('{"args": ["by-model"]}\n')
        out = self.run_entry("0")
        self.assertIn("by-model", out)
        self.assertIn(f"Config: {self.data}/config/strata-glm-ud-iq3_xxs.json", out)

    def test_setup_and_start_command_lines(self):
        out = self.run_entry("0", CONTEXT="131072", GPUS="0,1,2")
        setup = next(line for line in out.splitlines() if line.startswith("SETUP:"))
        self.assertIn("setup.py --setup --yes --model UD-IQ3_XXS", setup)
        self.assertIn(f"--data-dir {self.data}", setup)
        self.assertIn("--host 0.0.0.0", setup)
        self.assertIn("--port 8101", setup)
        self.assertIn("--no-browser --no-start", setup)
        self.assertIn("--context 131072", setup)
        self.assertIn("--gpus 0,1,2", setup)
        self.assertNotIn("--park-mib", setup)                          # unset: setup's own pick
        self.assertNotIn("--gguf-dir", setup)
        start = next(line for line in out.splitlines() if line.startswith("STARTED WITH:"))
        self.assertIn("ARGS: setup.py --port 8101 --gpus 0,1,2", start)
        self.assertTrue((self.opt / "strata-glm-ud-iq3_xxs.json").is_symlink())   # linked right after the setup

    def test_parking_ram_own_gguf_files_one_card_and_a_key(self):
        out = self.run_entry("0", PARK_MIB="4096", GGUF_DIR="/models", GPU="1", API_KEY="k1")
        self.assertIn("Setting up glm-ud-iq3_xxs from the GGUF files in /models", out)
        setup = next(line for line in out.splitlines() if line.startswith("SETUP:"))
        for part in ("--api-key k1", "--park-mib 4096", "--gguf-dir /models", "--gpu 1"):
            self.assertIn(part, setup)
        self.assertNotIn("--context", setup)                             # unset: setup's pick for the VRAM
        start = next(line for line in out.splitlines() if line.startswith("STARTED WITH:"))
        self.assertIn("ARGS: setup.py --port 8101 --gpu 1 --api-key k1", start)   # the key at every start

    def test_port_and_model_from_the_environment(self):
        (self.data / "config" / "strata-glm-iq4_xs.json").write_text('{"args": ["four-bit"]}\n')
        out = self.run_entry("0", MODEL="IQ4_XS", PORT="9000")
        self.assertIn(f"Config: {self.data}/config/strata-glm-iq4_xs.json", out)

    def configs(self):
        return (sorted(p.name for p in (self.data / "config").iterdir()),
                sorted(p.name for p in self.opt.glob("strata-glm-*.json")))

    def test_the_4_bit_by_setups_name_is_set_up_and_started(self):
        """MODEL=UD-IQ4_XS, the name setup.py gives the 4-bit size (its files in the repository), on a new volume."""
        out = self.run_entry("0", MODEL="UD-IQ4_XS")
        setup_line = next(line for line in out.splitlines() if line.startswith("SETUP:"))
        self.assertIn("setup.py --setup --yes --model UD-IQ4_XS", setup_line)
        self.assertIn("Setting up glm-ud-iq4_xs: downloading the model "
                      f"(~{setup.MODELS['UD-IQ4_XS']['download_gb']:.0f} GB", out)
        self.assertIn("STARTED WITH: {\"args\": [\"from-setup\"]}", out)
        self.assertEqual(self.configs(), (["strata-glm-ud-iq4_xs.json"], ["strata-glm-ud-iq4_xs.json"]))
        self.assertTrue((self.opt / "strata-glm-ud-iq4_xs.json").is_symlink())

    def test_a_volume_set_up_as_iq4_xs_is_started_by_either_4_bit_name(self):
        """A volume set up when the 4-bit size was IQ4_XS keeps strata-glm-iq4_xs.json, as setup.py keeps an install's
        names (config_tag): MODEL=UD-IQ4_XS starts it, with no setup pass (which would download the model again when
        its files are not found) and no second config beside it."""
        (self.data / "config" / "strata-glm-iq4_xs.json").write_text('{"args": ["four-bit-as-iq4_xs"]}\n')
        out = self.run_entry("0", MODEL="UD-IQ4_XS")
        self.assertNotIn("SETUP:", out)
        self.assertIn(f"Config: {self.data}/config/strata-glm-iq4_xs.json", out)
        self.assertIn("STARTED WITH: {\"args\": [\"four-bit-as-iq4_xs\"]}", out)
        self.assertEqual(self.configs(), (["strata-glm-iq4_xs.json"], ["strata-glm-iq4_xs.json"]))
        self.assertIn("STARTED WITH: {\"args\": [\"four-bit-as-iq4_xs\"]}", self.run_entry("0", MODEL="IQ4_XS"))

    def test_a_volume_set_up_as_ud_iq4_xs_is_started_by_the_earlier_name(self):
        (self.data / "config" / "strata-glm-ud-iq4_xs.json").write_text('{"args": ["four-bit"]}\n')
        out = self.run_entry("0", MODEL="IQ4_XS")
        self.assertNotIn("SETUP:", out)
        self.assertIn(f"Config: {self.data}/config/strata-glm-ud-iq4_xs.json", out)
        self.assertIn("STARTED WITH: {\"args\": [\"four-bit\"]}", out)
        self.assertEqual(self.configs(), (["strata-glm-ud-iq4_xs.json"], ["strata-glm-ud-iq4_xs.json"]))

    def test_reinstall_keeps_the_name_the_volume_has(self):
        """REINSTALL=1 with MODEL=UD-IQ4_XS on a volume set up as IQ4_XS: setup is given IQ4_XS, so its config keeps
        that name and replaces the one on the volume (not a second one the start would have to ask about)."""
        (self.data / "config" / "strata-glm-iq4_xs.json").write_text('{"args": ["before"]}\n')
        out = self.run_entry("1", MODEL="UD-IQ4_XS")
        setup_line = next(line for line in out.splitlines() if line.startswith("SETUP:"))
        self.assertIn("--model IQ4_XS ", setup_line)
        self.assertIn("STARTED WITH: {\"args\": [\"from-setup\"]}", out)
        self.assertEqual(self.configs(), (["strata-glm-iq4_xs.json"], ["strata-glm-iq4_xs.json"]))
        self.assertIn("from-setup", (self.data / "config" / "strata-glm-iq4_xs.json").read_text())

    def test_the_3_bit_size_is_setups(self):
        out = self.run_entry("0")
        self.assertIn("Setting up glm-ud-iq3_xxs: downloading the model "
                      f"(~{setup.MODELS['UD-IQ3_XXS']['download_gb']:.0f} GB", out)

    def entry(self, **extra):
        env = {k: v for k, v in os.environ.items() if k != "STRATA_API_KEY"}
        env.update(STRATA_DATA=str(self.data), MODEL="UD-IQ3_XXS", REINSTALL="0", **extra)
        r = subprocess.run([SH, str(self.script)], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r

    def test_no_api_key_on_every_interface_is_warned_at_each_start(self):
        """HOST=0.0.0.0 (the default: a published port reaches nothing else) with no key: whoever reaches the
        published port can use the model, so every start says so and names the publish that keeps it on this host."""
        cfg = self.data / "config" / "strata-glm-ud-iq3_xxs.json"
        cfg.write_text('{"args": ["x"], "host": "0.0.0.0"}\n')
        err = self.entry().stderr
        self.assertIn("WARNING: no API key", err)
        self.assertIn("-p 127.0.0.1:8101:8101", err)
        self.assertIn("-e API_KEY=", err)
        self.assertNotIn("no API key", self.entry(STRATA_API_KEY="k9").stderr)    # the server's own variable
        cfg.write_text('{"args": ["x"], "host": "0.0.0.0", "api_key": "k0"}\n')     # set up with a key
        self.assertNotIn("no API key", self.entry().stderr)
        cfg.write_text('{"args": ["x"], "host": "127.0.0.1"}\n')                    # nothing outside reaches it
        self.assertNotIn("no API key", self.entry().stderr)

    def test_api_key_counts_at_every_start(self):
        """API_KEY added after the first start (no REINSTALL): the start passes it on, so the server has it."""
        (self.data / "config" / "strata-glm-ud-iq3_xxs.json").write_text('{"args": ["x"], "host": "0.0.0.0"}\n')
        r = self.entry(API_KEY="k1")
        self.assertNotIn("no API key", r.stderr)
        start = next(line for line in r.stdout.splitlines() if line.startswith("STARTED WITH:"))
        self.assertIn("ARGS: setup.py --port 8101 --api-key k1", start)
        start = next(line for line in self.entry().stdout.splitlines() if line.startswith("STARTED WITH:"))
        self.assertNotIn("--api-key", start)                                          # none: nothing passed

    def test_the_dockerfile_publishes_on_this_host_only(self):
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("-p 127.0.0.1:8101:8101", text)
        self.assertNotIn("#     -p 8101:8101 \\", text)

    def test_an_unknown_model_stops(self):
        env = dict(os.environ, STRATA_DATA=str(self.data), MODEL="IQ2_XS", REINSTALL="0")
        r = subprocess.run([SH, str(self.script)], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 1)
        self.assertIn("UD-IQ3_XXS", r.stderr)
        self.assertIn("UD-IQ4_XS", r.stderr)

    def warning_for(self, flags):
        info = self.tmp / "cpuinfo"
        info.write_text(f"processor\t: 0\nflags\t\t: {flags}\n")
        (self.data / "config" / "strata-glm-ud-iq3_xxs.json").write_text('{"args": ["x"]}\n')
        env = dict(os.environ, STRATA_DATA=str(self.data), MODEL="UD-IQ3_XXS", REINSTALL="0", STRATA_CPUINFO=str(info))
        r = subprocess.run([SH, str(self.script)], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)     # a warning, never a stop (#1584)
        return r.stderr

    def test_a_cpu_without_avx2_is_warned_not_stopped(self):
        err = self.warning_for("fpu sse4_2 avx aes")
        self.assertIn("no AVX2", err)
        self.assertIn("docker build", err)

    def test_a_cpu_with_avx2_gets_no_warning(self):
        self.assertEqual(self.warning_for("fpu sse4_2 avx avx2 fma"), "")


if __name__ == "__main__":
    unittest.main()
