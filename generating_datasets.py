from BooleanNetwork import BooleanNetwork
import os


def create_MIR9_network():

    number_of_nodes = 6

    transitions = [
                   ([2], [0,1]),
                   ([3,5], [0,1,1,1]),
                   ([1,4], [1,0,0,0]),
                   ([0,4], [1,0,0,0]),
                   ([0,3], [1,0,0,0]),
                   ([0,4], [1,0,0,0]) 
                   ]
    model_MIR9 = BooleanNetwork(number_of_nodes, transitions)

    return model_MIR9

def generating_MIR9_datasets(model_MIR9):
    number_of_datapoints_options = [ i for i in range(3,40,3)]
    trajectory_length_options = [ i for i in range(3,40,3)]
    synchronous_options = [False, True]
    sampling_frequency_options = [1, 2, 3]
    starting_states = None #always start with random state
    
    count = 0
    if not os.path.exists('generated_datasets_from_part2'):
        os.mkdir('generated_datasets_from_part2')

    for number_of_datapoints in number_of_datapoints_options:
        for trajectory_length in trajectory_length_options:
            for synchronous in synchronous_options:
                for sampling_frequency in sampling_frequency_options:
                    dataset = model_MIR9.create_dataset( 
                              number_of_datapoints = number_of_datapoints,
                              trajectory_length = trajectory_length,
                              synchronous = synchronous,
                              sampling_frequency = sampling_frequency,
                              starting_states = None 
                              )
                    is_sync = 'Sync' if synchronous else 'Async'
                    filename = 'ds_NumPoints{}_Len{}_{}_Freq{}.txt'.format(number_of_datapoints, trajectory_length, is_sync, sampling_frequency)
                    model_MIR9.save_dataset(os.path.join('generated_datasets_from_part2', filename), dataset)

                    #print('Generated: ', filename) 
                    count += 1
    print('----DONE GENERATING DATASETS     total files generated = ', count)
                                                 

def main():
    # creating a chosen model
    model_MIR9 = create_MIR9_network()

    #saving model
    model_MIR9.save_network('model_MIR9.txt')

    # generating dataset/tajectories
    generating_MIR9_datasets(model_MIR9)


if __name__ == "__main__":
    main() 
