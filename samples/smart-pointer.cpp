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
#include <memory>
#include <numbers>

using namespace std::numbers; // std::numbers::pi

int main()
{
	sycl::queue queue{sycl::gpu_selector_v};

	constexpr utx::uc32 size = 6*6;
	auto ptr = std::make_shared<utx::fc32[]>(size);
	for (utx::uc32 i=0; i<size; i++)
		ptr[i] = -pi + i*pi/20;
	
	auto buffer = new sycl::buffer<utx::fc32, 2>{ptr, sycl::range<2>{6, 6}};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto acc = buffer->get_access<sycl::access_mode::read_write>(handler);
			auto lm = sycl::local_accessor<utx::fc32, 2>{sycl::range<2>{3,3*2}, handler};
			handler.parallel_for<class kn>(
				sycl::nd_range<2>{
					sycl::range<2>{6,6},
					sycl::range<2>{3,3}
				},
				[=] (sycl::nd_item<2> item)
				{
					utx::uc32 gid0 = item.get_global_id(0);
					utx::uc32 gid1 = item.get_global_id(1);
					utx::uc32 lid0 = item.get_local_id(0);
					utx::uc32 lid1 = item.get_local_id(1);

					utx::fc32 & lm0 = lm[lid0][lid1];
					utx::fc32 & lm1 = lm[lid0][lid1+1];

					utx::fc32 & rw = acc[gid0][gid1];

					lm0 = rw;
					sycl::group_barrier(item.get_group());

					lm1 = utx::sin(lm0);
					sycl::group_barrier(item.get_group());

					rw = lm1;
				}
			);
		}
	);

	delete buffer;

	for (utx::uc32 j=0; j<6; j++)
	{
		for (utx::uc32 i=0; i<6; i++)
		{
			utx::printnl(ptr[j*6+i], "");
		}
		utx::print();
	}
}

