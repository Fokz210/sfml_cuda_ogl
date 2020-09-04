#include <cmath>
#include <stdio.h>

#include <gl/glew.h>
#include <gl/GL.h>
#include <SFML/Graphics.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "color.h"
#include "sfml_context.h"

constexpr auto M_PI = 3.14159265359;

__global__ void render (sfml_device_context<color> context, size_t frames);
__host__   void render (sfml_host_context<color> context, size_t frames);
__device__ __host__ color tranform_hue (const color & in, float H);



int main ()
{
	sf::RenderWindow window (sf::VideoMode (1920, 1080), "cuda + sfml");

	int device = 0;

	cudaFree (0);
	cudaGetDevice (&device);
	cudaGLSetGLDevice (device);

	printf_s ("%d", device);

	if (glewInit () != GLEW_OK)
		throw std::runtime_error ("glew fucked up");

	sfml_device_context<color> context (window.getSize().x, window.getSize().y);

	dim3 threads_per_block(16, 16);
	dim3 num_blocks(window.getSize().x / threads_per_block.x, window.getSize().y / threads_per_block.y);

	size_t frames = 0;

	while (window.isOpen ())
	{
		frames++;

		sf::Event event;

		while (window.pollEvent (event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		render <<< num_blocks, threads_per_block >>> (context, frames);

		//render (context, frames);

		context.update ();
		window.draw (context);
		context.clear ();
		window.display ();
		window.clear ();
	}
}

int mmain ()
{
	sf::RenderWindow window (sf::VideoMode (1920, 1080), "cuda + sfml");

	if (glewInit () != GLEW_OK)
	{
		printf ("fuck");
		return -1;
	}
	const auto size = 1920u * 1080u * 4u;

	int device = 0;

	cudaFree (0);
	cudaGetDevice (&device);
	cudaGLSetGLDevice (device);

	GLuint pbo{0};
	cudaGraphicsResource_t resource{0};

	void * device_ptr = nullptr;

	glGenBuffers (1, &pbo);

	glBindBuffer (GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData (GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_STREAM_DRAW);

	checkCudaCall (cudaGraphicsGLRegisterBuffer (&resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
}

__global__ void render (sfml_device_context<color> context, size_t frames)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	auto gradient = color
	{
		255,
		255,
		255,
		255
	};

	context.buffer[y][x] = tranform_hue (gradient, static_cast<float> (frames) * 2);
}

__host__ void render (sfml_host_context<color> context, size_t frames)
{
	for (auto y = 0u; y < context.buffer.height (); y++)
		for (auto x = 0u; x < context.buffer.width (); x++)
		{
			auto gradient = color
			{
				255,
				255,
				255,
				255
			};

			context.buffer[y][x] = tranform_hue (gradient, static_cast<float> (frames % 100) / 100.f);
		}
}

__device__ __host__ color tranform_hue (const color & in, float H)
{
	float U = cos (H * M_PI / 180);
	float W = sin (H * M_PI / 180);

	color ret;
	ret.r = (.299 + .701 * U + .168 * W) * in.r
		+ (.587 - .587 * U + .330 * W) * in.g
		+ (.114 - .114 * U - .497 * W) * in.b;
	ret.g = (.299 - .299 * U - .328 * W) * in.r
		+ (.587 + .413 * U + .035 * W) * in.g
		+ (.114 - .114 * U + .292 * W) * in.b;
	ret.b = (.299 - .3 * U + 1.25 * W) * in.r
		+ (.587 - .588 * U - 1.05 * W) * in.g
		+ (.114 + .886 * U - .203 * W) * in.b;
	ret.a = in.a;

	return ret;
}
