utxcpp + sycl samples
================================================================================

sycl samples using utxcpp.

sycl
------------------------------

[sycl 2020](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)

utxcpp
------------------------------

[utxcpp](https://cppfx.xyz/utxcpp)

enum class sycl access_mode
------------------------------

sycl::access_mode

```c++
namespace sycl
{
	enum class access_mode
	{
		read,
		write,
		read_write,
		discard_write,		// deprecated in sycl 2020
		discard_read_write,		// deprecated in sycl 2020
		atomic		// deprecated in sycl 2020
	};
}

// sycl::access::mode is deprecated.
```

enum class target
------------------------------

sycl::target

```c++
namespace sycl
{
	enum class target
	{
		device,
		host_task,
		constant_buffer,		// deprecated
		local,		// deprecated
		host_buffer,		// deprecated
		global_buffer = device		// deprecated
	};
}

// sycl::access::target is deprecated.
```

barrier
------------------------------

Barrier primitives are made available to synchronize sets of work-items within individual groups. They are exposed through the group_barrier function.

```c++
sycl::group_barrier(__group);
```

SYCL Matrix Multiply Sample
------------------------------

```c++
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

	std::vector<utx::ic32>
		mat1 =
			{
				1,2,3,4,
				3,2,1,4,
				2,1,3,4,
				4,3,1,2
			},
		mat2 =
			{
				1,1,2,1,
				2,1,3,2,
				3,3,1,4,
				2,1,2,3
			},
		mat3(4*4)
	;

	auto buff1 = new sycl::buffer<utx::ic32, 2>{mat1.data(), sycl::range<2>{4,4}};
	auto buff2 = new sycl::buffer<utx::ic32, 2>{mat2.data(), sycl::range<2>{4,4}};
	auto buff3 = new sycl::buffer<utx::ic32, 2>{mat3.data(), sycl::range<2>{4,4}};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto acc1 = buff1->get_access<sycl::access::mode::read>(handler);
			auto acc2 = buff2->get_access<sycl::access::mode::read>(handler);
			auto acc3 = buff3->get_access<sycl::access::mode::write>(handler);
			auto lm = sycl::local_accessor<utx::ic32, 2>{sycl::range<2>{2,2}, handler};
			handler.parallel_for<class kernel_matrix_mul>(
				sycl::nd_range<2>{
					sycl::range<2>{4,4},
					sycl::range<2>{2,2}
				},
				[=] (sycl::nd_item<2> item)
				{
					utx::uc32 gid0 = item.get_global_id(0);
					utx::uc32 gid1 = item.get_global_id(1);

					utx::uc32 side = item.get_global_range(0); // 4
					//utx::uc32 side = item.get_global_range(1); // 4

					utx::uc32 lid0 = item.get_local_id(0);
					utx::uc32 lid1 = item.get_local_id(1);

					// name shorter alias
					utx::ic32 & lmr = lm[lid0][lid1];

					lmr = 0;
					item.barrier(sycl::access::fence_space::local_space);

					for (utx::uc32 ij=0; ij<side; ij++)
					{
						lmr += acc1[gid0][ij] * acc2[ij][gid1];
						item.barrier(sycl::access::fence_space::local_space);
					}

					acc3[gid0][gid1] = lmr;
				}
			);
		}
	);

	delete buff1;
	delete buff2;
	delete buff3;

	auto print_matrix = [] (const auto & mat)
	{
		for (utx::uc32 j=0; j<4; j++)
		{
			for (utx::uc32 i=0; i<4; i++)
			{
				utx::printnl(mat[j*4+i], "");
			}
			utx::print();
		}
	};

	print_matrix(mat1);
	utx::print("x");
	print_matrix(mat2);
	utx::print("=");
	print_matrix(mat3);
}
```

output:

```
1 2 3 4 
3 2 1 4 
2 1 3 4 
4 3 1 2 
x
1 1 2 1 
2 1 3 2 
3 3 1 4 
2 1 2 3 
=
22 16 19 29 
18 12 21 23 
21 16 18 28 
17 12 22 20 
```

utxcpp
------------------------------

[utxcpp](https://cppfx.xyz/utxcpp)

[utx::print](https://cppfx.xyz/utxcpp/utx_print.html)

[utx::print_all](https://cppfx.xyz/utxcpp/utx_print_all.html)

[utx::iota](https://cppfx.xyz/utxcpp/utx_iota.html)

[utx::uc8 utx::uc16 utx::uc32 utx::uc64 utx::ucmax](https://cppfx.xyz/utxcpp/class_types.html#class_types.role_cyan_utx_uc8_utx_uc16_utx_u)

[utx::ic8 utx::ic16 utx::ic32 utx::ic64 utx::icmax](https://cppfx.xyz/utxcpp/class_types.html#class_types.role_cyan_utx_ic8_utx_ic16_utx_i)

[utx::fc32 utx::fc64 utx::fc128 utx::fcmax](https://cppfx.xyz/utxcpp/class_types.html#class_types.role_cyan_utx_fc32_utx_fc64_utx_)

[utx::sqrt](https://cppfx.xyz/utxcpp/utx_sqrt.html)

[utx::cbrt](https://cppfx.xyz/utxcpp/utx_cbrt.html)

[utx::sin](https://cppfx.xyz/utxcpp/utx_sin.html)

[utx::cos](https://cppfx.xyz/utxcpp/utx_cos.html)

LICENSE
------------------------------

Boost Software License



