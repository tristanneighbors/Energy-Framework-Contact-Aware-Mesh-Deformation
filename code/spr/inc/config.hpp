#pragma once

/**
 * @file config.hpp
 * @brief Header-level project configuration switches.
 *
 * `config::DEBUG` follows the compiler's debug defines. `config::MATH_SAFETY`
 * controls optional checks and repairs for degenerate mathematical inputs; when
 * disabled, callers are responsible for satisfying each function's assumptions.
 */

namespace config {

#if defined(DEBUG) || defined(_DEBUG)
    inline constexpr bool DEBUG = true;
#else
    inline constexpr bool DEBUG = false;
#endif

#ifndef SPR_ENABLE_MATH_SAFETY
    inline constexpr bool MATH_SAFETY = true;
#else
    inline constexpr bool MATH_SAFETY = (SPR_ENABLE_MATH_SAFETY != 0);
#endif

}
