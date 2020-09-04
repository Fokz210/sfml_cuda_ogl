#pragma once

#include <SFML/Graphics.hpp>
#include "smart_buffer.h"

#include "cuda_check.h"

template <cuda_memory_type mode, class color_Ty>
class sfml_context
{
public:
	sfml_context (size_t width, size_t height);
	~sfml_context ();
	
	void clear () noexcept;
	void update () noexcept;
	void render (sf::RenderTarget & target);

	operator sf::Sprite();

	shared_buffer<mode, color_Ty> buffer;
protected:
	void gl_cuda_pbo_transfer (sf::Texture & texture, color_Ty * context, GLuint pbo);

	sf::Texture m_sfml_texture;
	GLuint m_pbo;
};

template<cuda_memory_type mode, class color_Ty>
inline sfml_context<mode, color_Ty>::sfml_context (size_t width, size_t height) : 
	m_sfml_texture (),
	buffer (width, height),
	m_pbo (0u)
{
	m_sfml_texture.create (width, height);
	glGenBuffers (1, &m_pbo);
}

template<cuda_memory_type mode, class color_Ty>
inline sfml_context<mode, color_Ty>::~sfml_context ()
{
	glBindBuffer (GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers (1, &m_pbo);
}

template<cuda_memory_type mode, class color_Ty>
inline void sfml_context<mode, color_Ty>::clear () noexcept
{
	buffer.clear ();
}

template<cuda_memory_type mode, class color_Ty>
inline void sfml_context<mode, color_Ty>::update () noexcept
{
	switch (mode)
	{
	case cuda_memory_type::host:

		m_sfml_texture.update (reinterpret_cast<sf::Uint8*>(buffer.get ()));
		break;

	case cuda_memory_type::device:
		gl_cuda_pbo_transfer (m_sfml_texture, buffer.get (), m_pbo);
		break;
	}
}

template<cuda_memory_type mode, class color_Ty>
inline void sfml_context<mode, color_Ty>::render (sf::RenderTarget & target)
{
	target.draw (sf::Sprite (m_sfml_texture));
}

template<cuda_memory_type mode, class color_Ty>
inline sfml_context<mode, color_Ty>::operator sf::Sprite ()
{
	return sf::Sprite (m_sfml_texture);
}

template<cuda_memory_type mode, class color_Ty>
inline void sfml_context<mode, color_Ty>::gl_cuda_pbo_transfer (sf::Texture & texture, color_Ty * context, GLuint pbo)
{
	const int x = static_cast<int>(texture.getSize ().x);
	const int y = static_cast<int>(texture.getSize ().y);

	cudaGraphicsResource_t res{0};

	glBindBuffer (GL_PIXEL_UNPACK_BUFFER, m_pbo);
	glBufferData (GL_PIXEL_UNPACK_BUFFER, 1920 * 1080 * 4, nullptr, GL_STREAM_DRAW);
	checkCudaCall (cudaGraphicsGLRegisterBuffer (&res, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

	void * ptr = nullptr;
	size_t buffsize = 0u;

	checkCudaCall (cudaGraphicsMapResources (1, &res));
	checkCudaCall (cudaGraphicsResourceGetMappedPointer (&ptr, &buffsize, res));
	checkCudaCall (cudaMemcpy (ptr, context, buffsize, cudaMemcpyDeviceToDevice));
	checkCudaCall (cudaGraphicsUnmapResources (1, &res));
	checkCudaCall (cudaGraphicsUnregisterResource (res));

	sf::Texture::bind (&texture);
	glTexImage2D (
		GL_TEXTURE_2D,
		0,
		GL_RGBA,
		x, y, 0,
		GL_RGBA,
		GL_UNSIGNED_BYTE,
		0x0
	);
}

template<class color_Ty>
using sfml_host_context = sfml_context<cuda_memory_type::host, color_Ty>;

template<class color_Ty>
using sfml_device_context = sfml_context<cuda_memory_type::device, color_Ty>;
