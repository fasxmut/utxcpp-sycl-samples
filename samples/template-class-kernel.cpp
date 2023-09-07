//
// Copyright (c) 2023 Fas Xmut (fasxmut at protonmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <sycl/sycl.hpp>
#include <utxcpp/core.hpp>
#include <utxcpp/algorithm.hpp>
#include <utxcpp/math.hpp>

template <typename data_type>
class utx_cbrt_kernel_class
{
public:
	using rw_accessor = sycl::accessor<data_type, 1, sycl::access_mode::read_write, sycl::target::device>;
private:
	rw_accessor acc;
public:
	utx_cbrt_kernel_class(rw_accessor acc):
		acc{acc}
	{
	}
	void operator()(sycl::nd_item<1> item) const
	{
		utx::uc32 gid = item.get_global_id(0);
		acc[gid()] = utx::cbrt(acc[gid()]);
	}
};

int main()
{
	sycl::queue queue;
	try
	{
		queue = sycl::queue{sycl::gpu_selector_v};
	}
	catch (const sycl::exception & err)
	{
		utx::printe("sycl::exception:", err.what());
		return 1;
	}

	std::vector<utx::fc32> vector(8);
	utx::iota(vector, 1);
	auto buff = new sycl::buffer<utx::fc32, 1>{vector};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto acc = buff->get_access<sycl::access_mode::read_write>(handler);
			handler.parallel_for<class utx_cbrt_kernel>(
				sycl::nd_range<1>{sycl::range<1>{vector.size()}, sycl::range<1>{vector.size()%2==0?2u:1u}},
				utx_cbrt_kernel_class{acc}
			);
		}
	);

	delete buff;

	utx::print_all(vector);
}

