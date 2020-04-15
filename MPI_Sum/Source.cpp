#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <mpi.h>
#include <queue>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    int rank, total;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    std::cout << "I am rank, " << rank << std::endl;
    if (rank == 0) {
		// Заполняем вектор с данными
        srand(time(0));
        std::vector<int> values;
        for (int i = 0; i < 20; ++i) {
            values.push_back(rand() % 10);
            std::cout << "Add element" << std::endl;
        }
        std::cout << "End of adding" << std::endl;
        std::cout << "Sum of elems: ";
        for (int i = 0; i < 20; ++i) {
            std::cout << values[i] << " ";
        }
        std::cout << std::endl;
        std::vector<int> answers;
        int current_pos = 0;
        int current_rank = 1;
        int flag = 1;
        do {
			// Обрабатываем задачу для каждого потока
            for (int i = 1; i < total; ++i) {
				// Если не осталось чисел для сложения в векторе 
                if (current_pos == values.size())
                {
                    values.clear();
                    break;
                }
				// Если осталось одно число для сложения в векторе
                if (current_pos == values.size()-1)
                {
                    answers.push_back(values[current_pos]);
                    values.clear();
                    current_pos = 0;
                    break;
                }
				// Если чисел больше одного, то пересылаем свободному потоку
                int nums[2];
                nums[0] = values[current_pos];
                nums[1] = values[(size_t)current_pos + 1];
                MPI_Send(&flag, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&nums, 2, MPI_INT, i, 1, MPI_COMM_WORLD);
                std::cout << "Send " << nums[0] << " " << nums[1] << " to " << i << " rank" << std::endl;
                current_rank = i+1;
                current_pos += 2;
            }
			// Ожидаем данные от потоков
            for (int i = 1; i < current_rank; ++i)
            {
                MPI_Status status;
                int val;
                MPI_Recv(&val, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
                std::cout << "Recieved from " << i << " number " << val << std::endl;
                answers.push_back(val);
            }
			// Если осталось всего 1 число, то получили ответ
            if (answers.size() == 1) {
                std::cout << "Answer: " << answers[0] << std::endl;
                break;
            }
			// Если чисел больше
            else
            {
				// Если чисел в векторе для сложения меньше 2ух, то просто очищаем вектор для сложения ( одно число уже перекинули ) и меняем местами вектора
                if (values.size() == 0 || values.size() == 1)
                {
                    std::cout << "Pack data and change to answers" << std::endl;
                    values = answers;
                    answers.clear();
                    current_pos = 0;
                }
            }
        } while (true);
        flag = 0;
		// Если получили ответ - подаём всем потокам сигнал о завершении работы
        for (int i = 1; i < total; ++i)
        {
            MPI_Send(&flag, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    }
    else {
        int input[2];
        int result;
        int have_value;
        MPI_Status status;
        while (true) {
			// Если приходит 1 - есть данные, если приходит 0 - завершаем работу
            MPI_Recv(&have_value, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            if (!have_value)
                break;
            MPI_Recv(&input, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            result = input[0] + input[1];
            MPI_Send(&result, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            
        }        
    }
    MPI_Finalize();
}