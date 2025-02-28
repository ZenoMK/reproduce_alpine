import os
import random
import argparse

def generate_random_list(num_nodes):
    """Generate a random list of integers with length between 5 and 100,
       followed by '%' and the reversed list."""
    length = 20
    rand_list = [random.randint(0, num_nodes - 1) for _ in range(length)]  # Generate random integers (0-1000)
    reversed_list = list(reversed(rand_list))
    return rand_list, reversed_list

def format_list(rand_list, reversed_list):
    """Format the list as a string with a '%' separator."""
    return " ".join(map(str, rand_list)) + " % " + " ".join(map(str, reversed_list)) + "\n"

def write_dataset(num_samples, file_name, num_nodes):
    """Generate and write multiple formatted list to a file."""
    with open(file_name, "w") as file:
        for _ in range(num_samples):
            rand_list, reversed_list = generate_random_list(num_nodes)
            file.write(format_list(rand_list, reversed_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random list and write them to files.')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--num_nodes', type=int, default=1000, help='Used for file path consistency')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Used for file naming consistency')

    args = parser.parse_args()

    folder_name = os.path.join(os.path.dirname(__file__), f'{args.num_nodes}_list')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    train_file = os.path.join(folder_name, f'train_{args.num_of_paths}.txt')
    test_file = os.path.join(folder_name, 'test.txt')

    # Generate and write datasets
    write_dataset(args.num_samples, train_file, args.num_nodes)
    write_dataset(args.num_samples // 5, test_file, args.num_nodes)  # Test set is smaller

    print(f"Generated {args.num_samples} training samples in {train_file}")
    print(f"Generated {args.num_samples // 5} test samples in {test_file}")
