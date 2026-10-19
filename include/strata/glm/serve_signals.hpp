// include/strata/glm/serve_signals.hpp - the signals of strata-glm in --serve mode.
//
// The server ends a serving engine itself: STOP and QUIT, after which the engine saves its expert profile
// (--expert-profile) and gives its memory back. It starts the engine in a session of its own, so Ctrl+C and the
// hang-up of a closed terminal reach only the server. A stop signal that still reaches a serving engine while that
// server runs - a systemd stop with the default KillMode=control-group sends SIGTERM to every process of the
// service, a `kill` - is therefore ignored (the log says so): the server's QUIT follows. An engine run by hand from
// a shell, or by a tool, is not in a session of its own: Ctrl+C and a `kill` end it as before.
//
// Without its server the engine still ends: when idle at the end of its input (the profile is saved), during a
// request at its next output line (SIGPIPE), on a stop signal, and while it still starts (placing the layers, the
// warm-up) at once when the server is gone.
#pragma once

#include <csignal>
#include <initializer_list>

#include <sys/prctl.h>
#include <unistd.h>

namespace strata::glm {

namespace serve_signals_detail {

inline pid_t server = 0;   ///< the process that started the engine: its parent at the start

inline void on_stop(int sig) {
    if (getppid() == server) {                     // the server runs: its QUIT ends the engine (async-signal-safe)
        char line[200];
        size_t n = 0;
        for (const char* part : {"strata-glm: ", sig == SIGINT ? "SIGINT" : sig == SIGHUP ? "SIGHUP" : "SIGTERM",
                                 " ignored while the server runs: the server ends the engine with QUIT (the expert "
                                 "profile is saved); kill -9 ends it at once\n"})
            for (; *part && n < sizeof line; ++part) line[n++] = *part;
        const ssize_t r = write(STDERR_FILENO, line, n);
        (void) r;
        return;
    }
    struct sigaction dfl {};                       // the server is gone: the signal's own way
    dfl.sa_handler = SIG_DFL;
    sigemptyset(&dfl.sa_mask);
    sigaction(sig, &dfl, nullptr);
    raise(sig);
}

}  // namespace serve_signals_detail

/// First in --serve mode, before the model is loaded: while it starts, the engine ends with the server that started
/// it (it has nothing to save yet), and a closed stdout ends it (SIGPIPE, also when the launcher ignored it).
inline void serve_signals_starting() {
    serve_signals_detail::server = getppid();
    std::signal(SIGPIPE, SIG_DFL);
    prctl(PR_SET_PDEATHSIG, SIGKILL);
}

/// Just before the serve loop says READY: the parent-death signal goes (it follows the thread that started the
/// engine, and a restart's is a request's thread, which ends), and SIGINT, SIGHUP and SIGTERM are ignored while the
/// server runs - only when the server started it in a session of its own. One ignored from the start (nohup's
/// SIGHUP) stays ignored.
inline void serve_signals_serving() {
    prctl(PR_SET_PDEATHSIG, 0);
    if (getsid(0) != getpid()) return;             // by hand from a shell, or a tool: Ctrl+C and kill end it
    struct sigaction sa {};
    sa.sa_handler = serve_signals_detail::on_stop;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    for (int sig : {SIGINT, SIGHUP, SIGTERM}) {
        struct sigaction old {};
        if (sigaction(sig, nullptr, &old) == 0 && old.sa_handler == SIG_IGN) continue;
        sigaction(sig, &sa, nullptr);
    }
}

}  // namespace strata::glm
