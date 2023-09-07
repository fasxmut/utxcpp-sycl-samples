//
// Copyright (c) 2023 Fas Xmut (fasxmut at protonmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Using std::simd, std::native_simd in sycl kernel.

#include <sycl/sycl.hpp>
#include <utxcpp/core.hpp>
#include <experimental/simd>
#include <utxcpp/flat.hpp>
#include <bit>
#include <utxcpp/algorithm.hpp>

namespace stdx = std::experimental;

int main()
{
	sycl::queue queue;
	try
	{
		queue = sycl::queue{sycl::gpu_selector_v};
	}
	catch (const sycl::exception & err)
	{
		try
		{
			queue = sycl::queue{sycl::cpu_selector_v};
		}
		catch (const sycl::exception & err)
		{
			utx::printe("sycl::exception:", err.what());
			return 1;
		}
	}

	using position_t = uflat::vector4uc32;
	using position_simd = stdx::native_simd<utx::u32>;

	constexpr utx::uc32 lw=2, lh=2;
	constexpr utx::uc32 gw=lw*2, gh=lh*8;
	std::vector<position_t> map(gw*gh);
	utx::iota(&map[0][0], &map[0][0]+map.size()*4, 1);

	auto print_map = [&]
	{
		for (utx::uc32 j=0; j<gh; j++)
		{
			for (utx::uc32 i=0; i<gw; i++)
			{
				const position_t & col = map[j*gw+i];
				utx::printnl('(', col[0], col[1], col[2], col[3], ")  ");
			}
			utx::print();
		}
	};

	utx::print("=>");
	print_map();

	auto buff = new sycl::buffer<position_t, 2>{map.data(), sycl::range<2>{gw, gh}};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto acc = buff->get_access<sycl::access_mode::read_write>(handler);
			auto lm = sycl::local_accessor<position_simd, 2>{sycl::range<2>{lw, lh}, handler};

			handler.parallel_for<class kn>(
				sycl::nd_range<2>{
					sycl::range<2>{gw, gh},
					sycl::range<2>{lw, lh}
				},
				[=] (sycl::nd_item<2> item)
				{
					utx::uc32 gid0 = item.get_global_id(0);
					utx::uc32 gid1 = item.get_global_id(1);
					utx::uc32 lid0 = item.get_local_id(0);
					utx::uc32 lid1 = item.get_local_id(1);

					position_t & pos = acc[gid0][gid1];
					position_simd & simd = lm[lid0][lid1];

					simd.copy_from(std::bit_cast<utx::u32 *>(&pos[0]), stdx::vector_aligned);
					sycl::group_barrier(item.get_group());

					simd *= simd;
					sycl::group_barrier(item.get_group());

					simd.copy_to(std::bit_cast<utx::u32 *>(&pos[0]), stdx::vector_aligned);
				}
			);
		}
	);

	delete buff;

	utx::print("=>");
	print_map();
}

/* output:
=>
( 1 2 3 4 )  ( 5 6 7 8 )  ( 9 10 11 12 )  ( 13 14 15 16 )  
( 17 18 19 20 )  ( 21 22 23 24 )  ( 25 26 27 28 )  ( 29 30 31 32 )  
( 33 34 35 36 )  ( 37 38 39 40 )  ( 41 42 43 44 )  ( 45 46 47 48 )  
( 49 50 51 52 )  ( 53 54 55 56 )  ( 57 58 59 60 )  ( 61 62 63 64 )  
( 65 66 67 68 )  ( 69 70 71 72 )  ( 73 74 75 76 )  ( 77 78 79 80 )  
( 81 82 83 84 )  ( 85 86 87 88 )  ( 89 90 91 92 )  ( 93 94 95 96 )  
( 97 98 99 100 )  ( 101 102 103 104 )  ( 105 106 107 108 )  ( 109 110 111 112 )  
( 113 114 115 116 )  ( 117 118 119 120 )  ( 121 122 123 124 )  ( 125 126 127 128 )  
( 129 130 131 132 )  ( 133 134 135 136 )  ( 137 138 139 140 )  ( 141 142 143 144 )  
( 145 146 147 148 )  ( 149 150 151 152 )  ( 153 154 155 156 )  ( 157 158 159 160 )  
( 161 162 163 164 )  ( 165 166 167 168 )  ( 169 170 171 172 )  ( 173 174 175 176 )  
( 177 178 179 180 )  ( 181 182 183 184 )  ( 185 186 187 188 )  ( 189 190 191 192 )  
( 193 194 195 196 )  ( 197 198 199 200 )  ( 201 202 203 204 )  ( 205 206 207 208 )  
( 209 210 211 212 )  ( 213 214 215 216 )  ( 217 218 219 220 )  ( 221 222 223 224 )  
( 225 226 227 228 )  ( 229 230 231 232 )  ( 233 234 235 236 )  ( 237 238 239 240 )  
( 241 242 243 244 )  ( 245 246 247 248 )  ( 249 250 251 252 )  ( 253 254 255 256 )  
=>
( 1 4 9 16 )  ( 25 36 49 64 )  ( 81 100 121 144 )  ( 169 196 225 256 )  
( 289 324 361 400 )  ( 441 484 529 576 )  ( 625 676 729 784 )  ( 841 900 961 1024 )  
( 1089 1156 1225 1296 )  ( 1369 1444 1521 1600 )  ( 1681 1764 1849 1936 )  ( 2025 2116 2209 2304 )  
( 2401 2500 2601 2704 )  ( 2809 2916 3025 3136 )  ( 3249 3364 3481 3600 )  ( 3721 3844 3969 4096 )  
( 4225 4356 4489 4624 )  ( 4761 4900 5041 5184 )  ( 5329 5476 5625 5776 )  ( 5929 6084 6241 6400 )  
( 6561 6724 6889 7056 )  ( 7225 7396 7569 7744 )  ( 7921 8100 8281 8464 )  ( 8649 8836 9025 9216 )  
( 9409 9604 9801 10000 )  ( 10201 10404 10609 10816 )  ( 11025 11236 11449 11664 )  ( 11881 12100 12321 12544 )  
( 12769 12996 13225 13456 )  ( 13689 13924 14161 14400 )  ( 14641 14884 15129 15376 )  ( 15625 15876 16129 16384 )  
( 16641 16900 17161 17424 )  ( 17689 17956 18225 18496 )  ( 18769 19044 19321 19600 )  ( 19881 20164 20449 20736 )  
( 21025 21316 21609 21904 )  ( 22201 22500 22801 23104 )  ( 23409 23716 24025 24336 )  ( 24649 24964 25281 25600 )  
( 25921 26244 26569 26896 )  ( 27225 27556 27889 28224 )  ( 28561 28900 29241 29584 )  ( 29929 30276 30625 30976 )  
( 31329 31684 32041 32400 )  ( 32761 33124 33489 33856 )  ( 34225 34596 34969 35344 )  ( 35721 36100 36481 36864 )  
( 37249 37636 38025 38416 )  ( 38809 39204 39601 40000 )  ( 40401 40804 41209 41616 )  ( 42025 42436 42849 43264 )  
( 43681 44100 44521 44944 )  ( 45369 45796 46225 46656 )  ( 47089 47524 47961 48400 )  ( 48841 49284 49729 50176 )  
( 50625 51076 51529 51984 )  ( 52441 52900 53361 53824 )  ( 54289 54756 55225 55696 )  ( 56169 56644 57121 57600 )  
( 58081 58564 59049 59536 )  ( 60025 60516 61009 61504 )  ( 62001 62500 63001 63504 )  ( 64009 64516 65025 65536 )  
*/
