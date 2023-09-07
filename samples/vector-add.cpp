//
// Copyright (c) 2023 Fas Xmut (fasxmut at protonmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <sycl/sycl.hpp>
#include <utxcpp/core.hpp>
#include <utxcpp/algorithm.hpp>

int main()
{
	sycl::queue queue{sycl::gpu_selector_v};

	constexpr utx::uc32 gsize = 9;
	constexpr utx::uc32 lsize = 3;
	std::vector<utx::ic32> add1(gsize);
	std::vector<utx::ic32> add2(gsize);
	std::vector<utx::ic32> result(gsize);
	utx::iota(add1, 1);
	utx::iota(add2, 37);

	auto buff1 = sycl::buffer<utx::ic32, 1>{add1};
	auto buff2 = sycl::buffer<utx::ic32, 1>{add2};
	auto buff3 = sycl::buffer<utx::ic32, 1>{result};
	
	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto acc1 = buff1.get_access<sycl::access::mode::read>(handler);
			auto acc2 = buff2.get_access<sycl::access::mode::read>(handler);
			auto acc3 = buff3.get_access<sycl::access::mode::write>(handler);
			handler.parallel_for<class kernel_vector_add>(
				sycl::nd_range<1>{
					sycl::range<1>{gsize},
					sycl::range<1>{lsize}
				},
				[=] (sycl::nd_item<1> item)
				{
					utx::uc32 gid = item.get_global_id();
					acc3[gid()] = acc1[gid()] + acc2[gid()];
				}
			);
		}
	);

	auto acc = buff3.get_host_access();
	utx::print_all(add1, add2, acc);
}

