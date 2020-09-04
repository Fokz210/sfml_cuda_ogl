#pragma once

#include <stdexcept>
#include <cuda_runtime.h>

enum class cuda_memory_type
{
	host,
	device
};

template<cuda_memory_type memory_type, class Ty>
class shared_buffer
{
public:
			   __host__ shared_buffer (size_t width, size_t height);
			   __host__ ~shared_buffer ();

			   __host__ shared_buffer (const shared_buffer & other);
			   __host__ shared_buffer & operator = (const shared_buffer & other);

	__device__ __host__ size_t width () const noexcept; 
	__device__ __host__ size_t height () const noexcept;

	__device__ __host__ constexpr cuda_memory_type type () const noexcept;

	__device__ __host__ Ty * operator [] (size_t y) noexcept;
	__device__ __host__ Ty * operator [] (size_t y) const noexcept;
	__device__ __host__ Ty * get () noexcept;

			   __host__ void clear () noexcept;

protected:
	__device__ __host__ shared_buffer (size_t width, size_t height, Ty * memory, size_t * counter);

			   __host__ void release ();

	size_t m_width, m_height;
	Ty * m_buffer;

	size_t * m_counter;
};

template<cuda_memory_type memory_type, class Ty>
__host__
inline shared_buffer<memory_type, Ty>::shared_buffer (size_t width, size_t height) :
	m_counter (new size_t(0)),
	m_width (width),
	m_height (height),
	m_buffer (nullptr)
{
	switch (memory_type)
	{
	case cuda_memory_type::host:
		m_buffer = new Ty[width * height];
		break;

	case cuda_memory_type::device:
#ifdef __CUDA_ARCH__
		m_buffer = reinterpret_cast<Ty*>(malloc (sizeof (Ty) * width * height));
#else
		cudaMalloc (&m_buffer, sizeof (Ty) * width * height);
#endif
		break;
	}

	clear ();
}

template<cuda_memory_type memory_type, class Ty>
__host__
inline shared_buffer<memory_type, Ty>::~shared_buffer ()
{
	if (*m_counter == 0)
		release ();
	else
		*m_counter -= 1;
}

template<cuda_memory_type memory_type, class Ty>
__host__
inline shared_buffer<memory_type, Ty>::shared_buffer (const shared_buffer & other) :
	m_counter (other.m_counter),
	m_buffer (other.m_buffer),
	m_width (other.m_width),
	m_height (other.m_height)
{
	(*m_counter)++;
}

template<cuda_memory_type memory_type, class Ty>
__host__
inline shared_buffer<memory_type, Ty> & shared_buffer<memory_type, Ty>::operator=(const shared_buffer & other)
{
	if (*m_counter == 0)
		release ();
	else
		*m_counter -= 1;

	m_buffer = other.m_buffer;
	m_width = other.m_width;
	m_height = other.m_height;
	m_counter = other.m_counter;

	(*m_counter)++;
}

template<cuda_memory_type memory_type, class Ty>
__device__ __host__
inline size_t shared_buffer<memory_type, Ty>::width () const noexcept
{
	return m_width;
}

template<cuda_memory_type memory_type, class Ty>
__device__ __host__
inline size_t shared_buffer<memory_type, Ty>::height () const noexcept
{
	return m_height;
}

template<cuda_memory_type memory_type, class Ty>
__device__ __host__
inline Ty * shared_buffer<memory_type, Ty>::get () noexcept
{
	return m_buffer;
}

template<cuda_memory_type memory_type, class Ty>
__host__
inline void shared_buffer<memory_type, Ty>::clear () noexcept
{
	switch (memory_type)
	{
	case cuda_memory_type::host:
		memset (m_buffer, 0x00, sizeof (Ty) * m_width * m_height);
		break;

	case cuda_memory_type::device:
#ifdef __CUDA_ARCH__
		memset (m_buffer, 0x00, sizeof (Ty) * m_width * m_height);
#else
		cudaMemset (m_buffer, 0x00, sizeof (Ty) * m_width * m_height);
#endif
		break;
	}
}

template<cuda_memory_type memory_type, class Ty>
__device__ __host__
inline shared_buffer<memory_type, Ty>::shared_buffer (size_t width, size_t height, Ty * memory, size_t * counter) :
	m_width (width),
	m_height (height),
	m_buffer (memory),
	m_counter (counter)
{
	(*counter)++;
}

template<cuda_memory_type memory_type, class Ty>
__host__
inline void shared_buffer<memory_type, Ty>::release ()
{
	switch (memory_type)
	{
	case cuda_memory_type::host:
		delete[] m_buffer;
		break;

	case cuda_memory_type::device:
		cudaFree (m_buffer);
#ifdef __CUDA_ARCH__
		free (m_buffer);
#else
		cudaFree (m_buffer);
#endif
		break;
	}

	delete m_counter;
}

template<cuda_memory_type memory_type, class Ty>
__device__ __host__
inline constexpr cuda_memory_type shared_buffer<memory_type, Ty>::type () const noexcept
{
	return memory_type;
}

template<cuda_memory_type memory_type, class Ty>
__device__ __host__
inline Ty * shared_buffer<memory_type, Ty>::operator[](size_t y) noexcept
{
	return m_buffer + m_width * (m_height - y - 1);
}

template<cuda_memory_type memory_type, class Ty>
__device__ __host__
inline Ty * shared_buffer<memory_type, Ty>::operator[](size_t y) const noexcept
{
	return m_buffer + m_width * (m_height - y - 1);
}

template<class Ty>
using host_smart_buffer = shared_buffer<cuda_memory_type::host, Ty>;

template<class Ty>
using device_smart_buffer = shared_buffer<cuda_memory_type::device, Ty>;

