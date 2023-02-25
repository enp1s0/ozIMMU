#pragma once
#include <stdexcept>

namespace mtk {
namespace ozimma {
namespace detail {
inline void print_not_implemented(
		const std::string file,
		const std::size_t line,
		const std::string func
		) {
	throw std::runtime_error("Not implemented (" + func + " in " + file + ", l." + std::to_string(line) + ")");
}
} // namespace detail
} // namespace ozimma
} // namespace mtk

#define OZIMMA_NOT_IMPLEMENTED mtk::ozimma::detail::print_not_implemented(__FILE__, __LINE__, __func__)
