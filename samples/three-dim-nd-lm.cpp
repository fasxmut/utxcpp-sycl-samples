//
// Copyright (c) 2023 Fas Xmut (fasxmut at protonmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// local memory on three-dimentional nd-range: use local memory on three-dimentional data of local range (group).
// We do utx::sqrt on it.

#include <utxcpp/core.hpp>
#include <utxcpp/math.hpp> // utx::sqrt
#include <utxcpp/algorithm.hpp> // utx::iota
#include <sycl/sycl.hpp>
#include <boost/assert.hpp>

int main()
{
	sycl::queue queue{sycl::gpu_selector_v};
	constexpr utx::uc32 ls0 = 2, ls1 = 2, ls2 = 2; // local range: 2x2x2
	constexpr utx::uc32 gs0 = ls0*2, gs1 = ls1*2, gs2 = ls2*3; // global range: 4x4x6

	std::vector<utx::fc32> src(gs0*gs1*gs2);
	utx::iota(src, 1);
	std::vector<utx::fc32> dst(src.size());

	auto src_buff = new sycl::buffer<utx::fc32, 3>{src.data(), sycl::range<3>{gs0, gs1, gs2}};
	auto dst_buff = new sycl::buffer<utx::fc32, 3>{dst.data(), sycl::range<3>{gs0, gs1, gs2}};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto src_acc = sycl::accessor{*src_buff, handler, sycl::read_only};
			auto dst_acc = sycl::accessor{*dst_buff, handler, sycl::write_only};
			constexpr utx::uc32 count = 2;
			// sycl local memory:
			//		Every group is granted with ls0 x ls1 x ls2*count elements on loal memory, (count==2),
			//		and every group size is ls0 x ls1 x ls2, so every item can use 2 elements on local memory.
			auto lm = sycl::local_accessor<utx::fc32, 3>{sycl::range<3>{ls0,ls1,ls2*count}, handler};
			handler.parallel_for<class three_dim_nd_lm_kernel>(
				sycl::nd_range<3>{
					sycl::range<3>{gs0, gs1, gs2},
					sycl::range<3>{ls0, ls1, ls2}
				},
				[=, count=count()] (sycl::nd_item<3> item)
				{
					utx::uc32 gid0 = item.get_global_id(0);
					utx::uc32 gid1 = item.get_global_id(1);
					utx::uc32 gid2 = item.get_global_id(2);

					// name shorter alias
					const utx::fc32 & src_ref = src_acc[gid0][gid1][gid2];
					utx::fc32 & dst_ref = dst_acc[gid0][gid1][gid2];

					utx::uc32 lid0 = item.get_local_id(0);
					utx::uc32 lid1 = item.get_local_id(1);
					utx::uc32 lid2 = item.get_local_id(2);

					utx::uc32 lsize2 = item.get_group(2); // We only need lsize2, do not need lsize0 and lsize1.

					BOOST_ASSERT(count==2);
					// name shorter alias
					utx::fc32 & lm0 = lm[lid0][lid1][lid2*count];
					utx::fc32 & lm1 = lm[lid0][lid1][lid2*count+1];

					// copy src to sycl local memory
					lm0 = src_ref;
					sycl::group_barrier(item.get_group());

					// computing on sycl local memory
					lm1 = utx::sqrt(lm0);
					sycl::group_barrier(item.get_group());

					// copy result from sycl local memory to buffer.
					dst_ref = lm1;
				}
			);
		}
	);

	delete src_buff;
	delete dst_buff;

	auto print_three_dim = [&gs0, &gs1, &gs2] (const std::vector<utx::fc32> & vector)
	{
		for (utx::uc32 k=0; k<gs0; k++)
		{
			for (utx::uc32 j=0; j<gs1; j++)
			{
				for (utx::uc32 i=0; i<gs2; i++)
				{
					utx::printnl(vector[k*gs1*gs2+j*gs2+i], "");
				}
				utx::print();
			}
			utx::print();
		}
	};

	utx::print("--------------------------------------------------------------------------------");
	utx::print("src vector: ----");
	print_three_dim(src);

	utx::print("--------------------------------------------------------------------------------");
	utx::print("dst_vector: ----");
	print_three_dim(dst);
}

/* output:
--------------------------------------------------------------------------------
src vector: ----
1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 
7.000000 8.000000 9.000000 10.000000 11.000000 12.000000 
13.000000 14.000000 15.000000 16.000000 17.000000 18.000000 
19.000000 20.000000 21.000000 22.000000 23.000000 24.000000 

25.000000 26.000000 27.000000 28.000000 29.000000 30.000000 
31.000000 32.000000 33.000000 34.000000 35.000000 36.000000 
37.000000 38.000000 39.000000 40.000000 41.000000 42.000000 
43.000000 44.000000 45.000000 46.000000 47.000000 48.000000 

49.000000 50.000000 51.000000 52.000000 53.000000 54.000000 
55.000000 56.000000 57.000000 58.000000 59.000000 60.000000 
61.000000 62.000000 63.000000 64.000000 65.000000 66.000000 
67.000000 68.000000 69.000000 70.000000 71.000000 72.000000 

73.000000 74.000000 75.000000 76.000000 77.000000 78.000000 
79.000000 80.000000 81.000000 82.000000 83.000000 84.000000 
85.000000 86.000000 87.000000 88.000000 89.000000 90.000000 
91.000000 92.000000 93.000000 94.000000 95.000000 96.000000 

--------------------------------------------------------------------------------
dst_vector: ----
1.000000 1.414214 1.732051 2.000000 2.236068 2.449490 
2.645751 2.828427 3.000000 3.162278 3.316625 3.464102 
3.605551 3.741657 3.872983 4.000000 4.123106 4.242640 
4.358899 4.472136 4.582576 4.690416 4.795832 4.898979 

5.000000 5.099020 5.196153 5.291503 5.385165 5.477225 
5.567764 5.656854 5.744563 5.830952 5.916080 6.000000 
6.082763 6.164414 6.244998 6.324555 6.403124 6.480741 
6.557439 6.633249 6.708204 6.782330 6.855655 6.928203 

7.000000 7.071068 7.141429 7.211102 7.280109 7.348469 
7.416199 7.483315 7.549834 7.615773 7.681146 7.745967 
7.810250 7.874008 7.937254 8.000000 8.062258 8.124039 
8.185352 8.246212 8.306623 8.366600 8.426149 8.485281 

8.544003 8.602325 8.660254 8.717798 8.774964 8.831760 
8.888195 8.944272 9.000000 9.055386 9.110434 9.165152 
9.219544 9.273619 9.327379 9.380832 9.433981 9.486834 
9.539392 9.591663 9.643651 9.695360 9.746794 9.797958 
*/

