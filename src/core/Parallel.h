#pragma once

#include <algorithm>
#include <thread>
#include <vector>

namespace vs {

/// Splits [0, count) into contiguous chunks, one per hardware thread, and calls
/// body(begin, end) for each chunk in parallel. Returns when all chunks are done.
/// `body` must not throw: an exception escaping a worker thread terminates the program.
template <typename Body>
void parallelFor(int count, Body&& body) {
    if (count <= 0) {
        return;
    }
    const int hardware = static_cast<int>(std::thread::hardware_concurrency());
    const int workers = std::clamp(hardware, 1, count);
    if (workers == 1) {
        body(0, count);
        return;
    }

    const int chunk = (count + workers - 1) / workers;
    std::vector<std::jthread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    for (int begin = 0; begin < count; begin += chunk) {
        const int end = std::min(count, begin + chunk);
        threads.emplace_back([&body, begin, end] { body(begin, end); });
    }
    // std::jthread joins on destruction.
}

} // namespace vs
