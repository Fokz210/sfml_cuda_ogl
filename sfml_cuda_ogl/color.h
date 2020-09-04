#pragma once
#include <cstdint>

union color
{
	struct
	{
		std::uint8_t r, g, b, a;
	};

	std::uint8_t arr[4];

	std::uint8_t & operator [] (size_t i) { return arr[i]; }
};