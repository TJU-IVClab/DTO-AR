#ifndef ORB_SLAM3_RUNTIME_PROFILE_H
#define ORB_SLAM3_RUNTIME_PROFILE_H

// Set to 0 (or define to 0 in the build) to compile out all probes.
#ifndef ORB_SLAM3_RUNTIME_PROFILE
#define ORB_SLAM3_RUNTIME_PROFILE 1
#endif

// Independent of profiling: 1 enables marker depth correction, 0 keeps raw depth.
#ifndef ORB_SLAM3_DEPTH_CORRECTION
#define ORB_SLAM3_DEPTH_CORRECTION 0
#endif

// Independent of profiling: 1 uses ellipse normals, 0 samples image gradients.
#ifndef ORB_SLAM3_ANALYTIC_BOUNDARY_NORMAL
#define ORB_SLAM3_ANALYTIC_BOUNDARY_NORMAL 1
#endif

#ifndef ORB_SLAM3_RUNTIME_PROFILE_DETAIL
#define ORB_SLAM3_RUNTIME_PROFILE_DETAIL 0
#endif

// Independent of profiling: enable the extra pre-WSBA chi2 diagnostic pass.
#ifndef ORB_SLAM3_WSBA_INITIAL_CHI
#define ORB_SLAM3_WSBA_INITIAL_CHI 0
#endif

#if ORB_SLAM3_RUNTIME_PROFILE
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <ostream>
#include <vector>

namespace ORB_SLAM3 { namespace RuntimeProfile {
using Clock = std::chrono::steady_clock;
enum Part { Preparation, Extraction, Matching, Pose, Keyframe, MapWait, Other, PartCount };
using Times = std::array<double, PartCount>;

#if ORB_SLAM3_RUNTIME_PROFILE_DETAIL
struct DetailRecord {
    std::array<double, 8> ms{};
    double total = 0;
    unsigned long keyframe = 0;
    int localKF = 0, fixedKF = 0, points = 0, edges = 0;
    bool graphReady = false, wroteBack = false;
    int optimizeReturn = -1; // -1: optimize() was not called.
    std::vector<int> iterationIds;
    std::vector<double> iterationMs;
};
#endif

struct FrameState {
    bool active = false;
    Part current = Other;
    Clock::time_point start, last;
    Times ms{};
};
inline FrameState& CurrentFrame() {
    static thread_local FrameState state;
    return state;
}
struct Store {
    std::atomic<bool> enabled{false};
    std::mutex mutex;
    std::vector<Times> frames;
    std::vector<double> totals, wsba;
#if ORB_SLAM3_RUNTIME_PROFILE_DETAIL
    std::vector<DetailRecord> extractionDetail, wsbaDetail;
#endif
};
inline Store& Data() { static Store data; return data; }
inline double Milliseconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
inline void BeginRun() {
    Store& data = Data();
    std::lock_guard<std::mutex> lock(data.mutex);
    data.frames.clear(); data.totals.clear(); data.wsba.clear();
#if ORB_SLAM3_RUNTIME_PROFILE_DETAIL
    data.extractionDetail.clear(); data.wsbaDetail.clear();
#endif
    data.enabled.store(true);
}
inline void BeginFrame(Clock::time_point start) {
    FrameState& state = CurrentFrame();
    state = FrameState();
    state.active = true;
    state.start = state.last = start;
}
inline void Charge(Clock::time_point now) {
    FrameState& state = CurrentFrame();
    state.ms[state.current] += Milliseconds(state.last, now);
    state.last = now;
}
// Category switches charge exclusive time, including nested scopes and early returns.
class Scope {
    bool active;
    Part previous;
public:
    explicit Scope(Part part) : active(CurrentFrame().active), previous(Other) {
        if (active) {
            Charge(Clock::now());
            previous = CurrentFrame().current;
            CurrentFrame().current = part;
        }
    }
    void Stop() {
        if (active) {
            Charge(Clock::now());
            CurrentFrame().current = previous;
            active = false;
        }
    }
    ~Scope() { Stop(); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
inline void EndFrame(Clock::time_point end) {
    FrameState& state = CurrentFrame();
    Charge(end);
    state.active = false;
    Store& data = Data();
    std::lock_guard<std::mutex> lock(data.mutex);
    data.frames.push_back(state.ms);
    data.totals.push_back(Milliseconds(state.start, end));
}
class WsbaScope {
    bool active;
    Clock::time_point start;
public:
    WsbaScope() : active(Data().enabled.load()) {
        if (active) start = Clock::now();
    }
    ~WsbaScope() {
        if (!active) return;
        const double elapsed = Milliseconds(start, Clock::now());
        Store& data = Data();
        std::lock_guard<std::mutex> lock(data.mutex);
        data.wsba.push_back(elapsed);
    }
    WsbaScope(const WsbaScope&) = delete;
    WsbaScope& operator=(const WsbaScope&) = delete;
};
#if ORB_SLAM3_RUNTIME_PROFILE_DETAIL
// Sequential stages partition one invocation, including early returns and cleanup.
class DetailScope {
    bool backend, active;
    int stage = 0;
    Clock::time_point start, last;
public:
    DetailRecord record;
    explicit DetailScope(bool isBackend) : backend(isBackend),
        active(isBackend ? Data().enabled.load() : CurrentFrame().active) {
        if (active) start = last = Clock::now();
    }
    void Next(int next) {
        if (!active) return;
        const auto now = Clock::now();
        record.ms[stage] += Milliseconds(last, now);
        last = now;
        stage = next;
    }
    ~DetailScope() {
        if (!active) return;
        Next(stage);
        record.total = Milliseconds(start, last);
        Store& data = Data();
        std::lock_guard<std::mutex> lock(data.mutex);
        (backend ? data.wsbaDetail : data.extractionDetail).push_back(record);
    }
    DetailScope(const DetailScope&) = delete;
    DetailScope& operator=(const DetailScope&) = delete;
};
#endif
inline void PrintRow(std::ostream& out, const char* name, std::vector<double> values) {
    out << std::left << std::setw(29) << name << std::right << std::setw(9) << values.size();
    if (values.empty()) { out << "  n/a\n"; return; }
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const double median = n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
    out << std::setw(12) << values.front() << std::setw(12) << median
        << std::setw(12) << sum / n << std::setw(12) << values[static_cast<std::size_t>(std::ceil(.95 * n)) - 1]
        << std::setw(12) << values.back() << std::setw(15) << sum << '\n';
}
#if ORB_SLAM3_RUNTIME_PROFILE_DETAIL
inline void PrintDetail(std::ostream& out, const char* title,
    const std::vector<DetailRecord>& records, const char* const* names, int stages) {
    out << '\n' << title << " (ms/call; subdivisions, do not add to parent rows)\n";
    for (int stage = 0; stage < stages; ++stage) {
        std::vector<double> values;
        for (const auto& record : records) values.push_back(record.ms[stage]);
        PrintRow(out, names[stage], values);
    }
    std::vector<double> totals;
    double error = 0;
    for (const auto& record : records) {
        totals.push_back(record.total);
        error = (std::max)(error, std::abs(std::accumulate(record.ms.begin(), record.ms.end(), 0.0) - record.total));
    }
    PrintRow(out, "Detail measured total", totals);
    out << "Max |sum(stages) - detail total|: " << error << " ms\n";
}
#endif
// Call only after LocalMapping has stopped, so its last invocation is included.
inline void Print(std::ostream& out) {
    Store& data = Data();
    data.enabled.store(false);
    std::lock_guard<std::mutex> lock(data.mutex);
    const auto flags = out.flags();
    const auto precision = out.precision();
    out << std::fixed << std::setprecision(4)
        << "\n[Runtime profile] OFFLINE frontend; all times in ms\n"
        << "Exclusive per-frame categories; inactive categories contribute zero.\n"
        << std::left << std::setw(29) << "Category" << std::right << std::setw(9) << "N"
        << std::setw(12) << "min" << std::setw(12) << "median" << std::setw(12) << "mean"
        << std::setw(12) << "P95" << std::setw(12) << "max" << std::setw(15) << "sum" << '\n';
    const char* names[] = {"Data preparation", "Fiducial extraction", "Topology matching",
        "Pose / local refinement", "Keyframe preparation", "Map-update lock wait", "Other / bookkeeping"};
    for (int part = 0; part < PartCount; ++part) {
        std::vector<double> values;
        values.reserve(data.frames.size());
        for (const Times& frame : data.frames) values.push_back(frame[part]);
        PrintRow(out, names[part], values);
    }
    PrintRow(out, "Frontend total (ttrack)", data.totals);
    double maxError = 0;
    for (std::size_t i = 0; i < data.frames.size(); ++i)
        maxError = std::max(maxError, std::abs(std::accumulate(data.frames[i].begin(), data.frames[i].end(), 0.0) - data.totals[i]));
    out << "Max |sum(parts) - total|: " << maxError << " ms\n"
        << "Async backend (per invocation, NOT added to frontend total):\n";
    PrintRow(out, "WSBA full call", data.wsba);
    out << "WSBA includes calls returning early or interrupted; N is invocation count.\n";
#if ORB_SLAM3_RUNTIME_PROFILE_DETAIL
    const char* extractionNames[] = {"Blur + Canny", "Contours", "Ellipse fitting",
        "Ellipse merging", "Undistort + gradients", "Validation + subpixel", "Cleanup"};
    PrintDetail(out, "Fiducial extraction detail", data.extractionDetail, extractionNames, 7);
    const char* wsbaNames[] = {"Window collection", "Graph construction", "Solver initialization",
        "Solver optimize", "Outliers + diagnostics", "Map lock wait", "Result writeback", "Cleanup"};
    PrintDetail(out, "WSBA detail", data.wsbaDetail, wsbaNames, 8);
    out << "WSBA workload per call (localKF includes any fixed origin in the local set):\n"
        << "call KF localKF fixedKF MPs edges graph_ready writeback total_ms"
        << " window_ms graph_ms init_ms solve_ms outlier_ms lock_ms writeback_ms cleanup_ms"
        << " optimize_return observed_iterations\n";
    for (std::size_t i = 0; i < data.wsbaDetail.size(); ++i) {
        const auto& r = data.wsbaDetail[i];
        out << i + 1 << ' ' << r.keyframe << ' ' << r.localKF << ' ' << r.fixedKF
            << ' ' << r.points << ' ' << r.edges << ' ' << r.graphReady << ' '
            << r.wroteBack << ' ' << r.total;
        for (double ms : r.ms) out << ' ' << ms;
        out << ' ' << r.optimizeReturn << ' ' << r.iterationMs.size() << '\n';
    }
    out << "WSBA iterations (pre/post callbacks; includes LM retries within each iteration):\n"
        << "call KF iteration_index elapsed_ms\n";
    std::vector<double> allIterations;
    for (std::size_t i = 0; i < data.wsbaDetail.size(); ++i) {
        const auto& r = data.wsbaDetail[i];
        for (std::size_t j = 0; j < r.iterationMs.size(); ++j) {
            out << i + 1 << ' ' << r.keyframe << ' ' << r.iterationIds[j]
                << ' ' << r.iterationMs[j] << '\n';
            allIterations.push_back(r.iterationMs[j]);
        }
    }
    out << "Iteration summary (N = observed iterations; ms):\n";
    PrintRow(out, "WSBA iteration", allIterations);
    out << "optimize_return=-1: not called; return=0 can indicate zero iterations or failure.\n"
        << "Observed iterations count post callbacks, not accepted LM updates; writeback is not convergence.\n";
#endif
    out.flags(flags); out.precision(precision);
}
} }
#endif
#endif
