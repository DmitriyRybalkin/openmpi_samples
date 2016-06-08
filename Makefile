examples: dot_product.c
	mpicc dot_product.c -o dot_product
	mpicc circuit_satisfiability.c -o circuit_satisfiability
	mpicc circuit_satisfiability_v2.c -o circuit_satisfiability_v2
	mpicc circuit_satisfiability_v3.c -o circuit_satisfiability_v3
	mpicc sieve_of_eratosthenes.c -o sieve_of_eratosthenes -lm
	mpicc floyd_algorithm.c -o floyd_algorithm -lm
	mpicc matrix_vector_multiplication.c -o matrix_vector_multiplication -lm
	mpicc matrix_vector_multiplication_v2.c -o matrix_vector_multiplication_v2 -lm
	mpicc document_classification.c -o document_classification -lm
	gcc -fopenmp compute_pi.cpp -o compute_pi -lstdc++
	gcc -fopenmp matrix_product.cpp -o matrix_product -lstdc++
clean:
	rm -f dot_product circuit_satisfiability circuit_satisfiability_v2 circuit_satisfiability_v3 sieve_of_eratosthenes floyd_algorithm matrix_vector_multiplication matrix_vector_multiplication_v2 document_classification compute_pi matrix_product