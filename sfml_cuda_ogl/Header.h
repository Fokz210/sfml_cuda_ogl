#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

template <class T>
class matrix
{
public:
	__host__ __device__  matrix(size_t rows_num, size_t collums_num);

	__host__ __device__  matrix(const matrix<T> & other);
	__host__ __device__  matrix(matrix<T> && other);
	__host__ __device__  matrix<T> & operator = (const matrix<T> & rhs);
	__host__ __device__  matrix<T> & operator = (matrix<T> && rhs);

	__host__ __device__  matrix<T> & operator = (std::initializer_list<T> list);

	__host__ __device__  T * operator [] (size_t i);
	__host__ __device__  T * operator [] (size_t i) const;

	__host__ __device__  void swap(matrix<T> & other);

	 size_t rows, columns;

protected:
	std::unique_ptr<T[]> m_data;
};

template <class T>
__host__
std::ostream & operator << (std::ostream & lhs, const matrix<T> & rhs);

template <class T>
__host__
std::istream & operator >> (std::istream & lhs, const matrix<T> & rhs);


template <class T>
__host__ __device__
matrix<T> operator * (const matrix<T> & lhs, const matrix<T> & rhs);


template<class T>
__host__ __device__
matrix<T>::matrix(size_t rows_num, size_t collums_num) :
	rows(rows_num),
	columns(collums_num),
	m_data(new T[rows_num * collums_num])
{
}

template<class T>
__host__ __device__
matrix<T>::matrix(const matrix<T> & other) :
	columns(other.columns),
	rows(other.rows),
	m_data(new T[other.rows * other.columns])
{
	for (auto i = 0u; i < rows * columns; i++)
		m_data[i] = other.m_data[i];
}

template<class T>
__host__ __device__
matrix<T>::matrix(matrix<T> && other) :
	columns(),
	rows(),
	m_data()
{
	swap(other);
}

template<class T>
__host__ __device__
matrix<T> & matrix<T>::operator=(const matrix<T> & rhs)
{
	matrix<T> temp(rhs);
	swap(temp);

	return *this;
}

template<class T>
__host__ __device__
matrix<T> & matrix<T>::operator=(matrix<T> && rhs)
{
	swap(rhs);

	return *this;
}

template<class T>
__host__ __device__
matrix<T> & matrix<T>::operator=(std::initializer_list<T> list)
{
	auto i = 0u;

	for (auto && el : list)
		m_data[i++] = el;

	return *this;
}

template<class T>
__host__ __device__
T * matrix<T>::operator[](size_t i)
{
	return m_data.get() + columns * i;
}

template<class T>
__host__ __device__
T * matrix<T>::operator[](size_t i) const
{
	return m_data.get() + columns * i;
}

template<class T>
__host__ __device__
void matrix<T>::swap(matrix<T> & other)
{
	std::swap(this->columns, other.columns);
	std::swap(this->rows, other.rows);
	std::swap(this->m_data, other.m_data);
}

template <class T>
__host__
std::ostream & operator<<(std::ostream & lhs, const matrix<T> & rhs)
{
	for (auto i = 0u; i < rhs.rows; i++)
	{
		for (auto j = 0u; j < rhs.columns; j++)
			lhs << rhs[i][j] << " ";
		if (i != rhs.rows - 1) lhs << std::endl;
	}

	return lhs;
}

template<class T>
__host__
std::istream & operator>>(std::istream & lhs, const matrix<T> & rhs)
{
	for (auto i = 0u; i < rhs.rows * rhs.columns; i++)
		lhs >> rhs[0][i];

	return lhs;
}

template<class T>
__host__ __device__ 
matrix<T> operator*(const matrix<T> & lhs, const matrix<T> & rhs)
{
	matrix<T> res(lhs.rows, rhs.columns);

	for (auto Si = 0u; Si < res.rows; Si++)
		for (auto Sj = 0u; Sj < res.columns; Sj++)
		{
			res[Si][Sj] = T();

			for (auto i = 0u; i < lhs.columns; i++)
			{
				res[Si][Sj] += lhs[Si][i] * rhs[i][Sj];
			}
		}

	return res;
}