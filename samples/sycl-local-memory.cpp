//
// Copyright (c) 2023 Fas Xmut (fasxmut at protonmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <sycl/sycl.hpp>
#include <utxcpp/core.hpp>
#include <utxcpp/math.hpp>
#include <utxcpp/algorithm.hpp>

int main()
{
	sycl::device device{sycl::gpu_selector_v};
	auto lmsize = device.get_info<sycl::info::device::local_mem_size>();
	utx::print("max local memory size:", lmsize);
	sycl::queue queue{device};
	
	constexpr utx::uc32 gsize = 8; // global range will be 8x8
	constexpr utx::uc32 lsize = 2; // local range will be 2x2

	std::vector<utx::fc32> vector(gsize*gsize);
	utx::iota(vector, 1.0f);

	utx::print("vector:");
	for (utx::uc32 j=0; j<gsize; j++)
	{
		for (utx::uc32 i=0; i<gsize; i++)
		{
			utx::printnl(vector[j*gsize+i], "");
		}
		utx::print();
	}

	auto buffer = sycl::buffer<utx::fc32, 2>{vector.data(), sycl::range<2>{gsize, gsize}};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto acc = buffer.get_access<sycl::access_mode::read_write>(handler);
			auto lm = sycl::local_accessor<utx::fc32, 2>{sycl::range<2>{lsize*1, lsize*1}, handler};
			handler.parallel_for<class lm_kernel>(
				sycl::nd_range<2>{
					sycl::range<2>{gsize, gsize},
					sycl::range<2>{lsize, lsize}
				},
				[=] (sycl::nd_item<2> item)
				{
					utx::uc32 gid0 = item.get_global_id(0);
					utx::uc32 gid1 = item.get_global_id(1);
					utx::uc32 lid0 = item.get_local_id(0);
					utx::uc32 lid1 = item.get_local_id(1);
					
					// name shorter alias
					utx::fc32 & lm0 = lm[lid0][lid1];
					utx::fc32 & gm = acc[gid0][gid1];

					// copy to sycl local memory
					lm0 = gm;
					sycl::group_barrier(item.get_group());

					// computing on local memory
					lm0 = utx::sqrt(lm0);
					sycl::group_barrier(item.get_group());

					// copy from local memory to global memory
					gm = lm0;
				}
			);
		}
	);
	auto acc = buffer.get_host_access();

	utx::print("after utx::sqrt:");
	for (utx::uc32 j=0; j<gsize; j++)
	{
		for (utx::uc32 i=0; i<gsize; i++)
		{
			utx::printnl(acc[j][i], "");
		}
		utx::print();
	}
}

