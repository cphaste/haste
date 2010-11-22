/* The MIT License:

Copyright (c) 2009-2010 Ivan Gagis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE. */

// ting 0.4.2
// Homepage: http://code.google.com/p/ting

/**
 * @file utils.hpp
 * @author Ivan Gagis <igagis@gmail.com>
 * @brief Utility functions and classes.
 */

#pragma once

//#ifdef _MSC_VER //If Microsoft C++ compiler
//#pragma warning(disable:4290)
//#endif

#include <vector>

#include "debug.hpp" //debugging facilities
#include "types.hpp"
#include "Thread.hpp"

//define macro used to align structures in memory
#ifdef _MSC_VER //If Microsoft C++ compiler
#define M_DECLARE_ALIGNED(x)
#define M_DECLARE_ALIGNED_MSVC(x) __declspec(align(x))

#elif defined(__GNUG__)//GNU g++ compiler
#define M_DECLARE_ALIGNED(x) __attribute__ ((aligned(x)))
#define M_DECLARE_ALIGNED_MSVC(x)

#else
#error "unknown compiler"
#endif


namespace ting{



/**
 * @brief Exchange two values.
 * Note, that it uses temporary variable to switch the values. Thus, the value should
 * have proper copy constructor defined.
 * @param a - reference to value a.
 * @param b - reference to value b.
 */
template <class T> inline void Exchange(T &a, T &b){
	T tmp = a;
	a = b;
	b = tmp;
}



#ifndef M_DOC_DONT_EXTRACT //for doxygen
//quick exchange two unsigned 32bit integers
template <> inline void Exchange<u32>(u32& x, u32& y){
//	TRACE(<<"Exchange<u32>(): invoked"<<std::endl)
	//NOTE: Do not make y^=x^=y^=x;
	//Some compilers (e.g. gcc4.1) may generate incorrect code
	y ^= x;
	x ^= y;
	y ^= x;
}
#endif



#ifndef M_DOC_DONT_EXTRACT //for doxygen
//quick exchange two floats
template <> inline void Exchange<float>(float& x, float& y){
//	TRACE(<<"Exchange<float>(): invoked"<<std::endl)
	Exchange<u32>(reinterpret_cast<u32&>(x), reinterpret_cast<u32&>(y));
}
STATIC_ASSERT(sizeof(float) == sizeof(u32))
#endif



/**
 * @brief Clamp value top.
 * This inline template function can be used to clamp the top of the value.
 * Example:
 * @code
 * int a = 30;
 *
 * //Clamp to 40. Value of 'a' remains unchanged,
 * //since it is already less than 40.
 * ting::ClampTop(a, 40);
 * std::cout << a << std::endl;
 *
 * //Clamp to 27. Value of 'a' is changed to 27,
 * //since it is 30 which is greater than 27.
 * ting::ClampTop(a, 27);
 * std::cout << a << std::endl;
 * @endcode
 * As a result, this will print:
 * @code
 * 30
 * 27
 * @endcode
 * @param v - reference to the value which top is to be clamped.
 * @param top - value to clamp the top to.
 */
template <class T> inline void ClampTop(T& v, const T top){
	if(v > top){
		v = top;
	}
}


/**
 * @brief Clamp value bottom.
 * Usage is analogous to ting::ClampTop.
 * @param v - reference to the value which bottom is to be clamped.
 * @param bottom - value to clamp the bottom to.
 */
template <class T> inline void ClampBottom(T& v, const T bottom){
	if(v < bottom){
		v = bottom;
	}
}



/**
 * @brief convert byte order of 16 bit value to network format.
 * @param value - the value.
 * @param out_buf - pointer to the 2 byte buffer where the result will be placed.
 */
inline void ToNetworkFormat16(u16 value, u8* out_buf){
	*reinterpret_cast<u16*>(out_buf) = value;//assume little-endian
}



/**
 * @brief convert byte order of 32 bit value to network format.
 * @param value - the value.
 * @param out_buf - pointer to the 4 byte buffer where the result will be placed.
 */
inline void ToNetworkFormat32(u32 value, u8* out_buf){
	*reinterpret_cast<u32*>(out_buf) = value;//assume little-endian
}



/**
 * @brief Convert 16 bit value from network byte order to native byte order.
 * @param buf - pointer to buffer containig 2 bytes to convert from network format.
 * @return 16 bit unsigned integer converted from network byte order to native byte order.
 */
inline u16 FromNetworkFormat16(const u8* buf){
	return *reinterpret_cast<const u16*>(buf);//assume little-endian
}



/**
 * @brief Convert 32 bit value from network byte order to native byte order.
 * @param buf - pointer to buffer containig 4 bytes to convert from network format.
 * @return 32 bit unsigned integer converted from network byte order to native byte order.
 */
inline u32 FromNetworkFormat32(const u8* buf){
	return *reinterpret_cast<const u32*>(buf);//assume little-endian
}



/**
 * @brief Maximal value of unsigned integer type.
 * @return Maximal value an unsigned integer type can represent on the current platform.
 */
inline unsigned DMaxUint(){
	return unsigned(-1);
}



/**
 * @brief Maximal value of integer type.
 * @return Maximal value an integer type can represent on the current platform.
 */
inline int DMaxInt(){
	return int(DMaxUint() >> 1);
}



/**
 * @brief Minimal value of integer type.
 * @return Minimal value an integer type can represent on the current platform.
 */
inline int DMinInt(){
	return ~DMaxInt();
}



}//~namespace ting

