//
// Copyright (c) 2023 Fas Xmut (fasxmut at protonmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// buffer.get_host_access()
// utx::meric_static_cast<type>(value)

#include <sycl/sycl.hpp>
#include <utxcpp/core.hpp>

int main()
{
	bool exception = false;
	sycl::queue queue{
		sycl::gpu_selector_v,
		[&exception] (sycl::exception_list elist)
		{
			for (std::exception_ptr eptr: elist)
			{
				try
				{
					std::rethrow_exception(eptr);
				}
				catch (const std::exception & err)
				{
					utx::printe("sycl exception:", err.what());
					exception = true;
				}
			}
		}
	};
	if (exception)
		return 1;
	
	sycl::buffer<utx::uc16, 1> buffer{sycl::range<1>{64}};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto acc = buffer.get_access<sycl::access_mode::write>(handler);
			handler.parallel_for<class get_kernel_nu>(
				sycl::range<1>{64},
				[=] (sycl::id<1> id)
				{
					acc[id] = utx::meric_static_cast<utx::uc16>(id[0]);
				}
			);
		}
	);
	utx::print_all(buffer.get_host_access());
}

