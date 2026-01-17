import os
import glob
import subprocess
import re
import sys


from EvaluationMetrics import (
    compute_metrics,             
    adjacency_matrix_from_file,
    adjacency_matrix_from_sif
)

#------------CONFIG------------
CURRENT_DIR = os.getcwd()
GROUND_TRUTH_FILE = os.path.join(CURRENT_DIR, 'model_MIR9.txt')
TABLES_DIR = os.path.join(CURRENT_DIR, 'tables')
RECONSTRUCTED_MIR9 = os.path.join(CURRENT_DIR, 'reconstructed_MIR9')
DATASETS_DIR  = os.path.join(CURRENT_DIR, 'generated_datasets_from_part2')

def ensure_directory(dir):
    """
    if 'dir' doesn't exists, it created directory 'dir' 
    """

    if not os.path.exists(dir):
        os.mkdir(dir)


def run_bnfinder(dataset_path, output_path):
    """
    running BNFinder2 for one dataset with MDL scoring function
    """
    bnf_path = os.path.abspath('bnf')
    cmd = 'python "{}" -s MDL -e "{}" -n "{}"'.format(bnf_path, dataset_path, output_path)

    try: 
        os.system(cmd)
        return True
    except:
        print("Couldn't run bnf for ", dataset_path)
        return False



def main():
    ensure_directory(RECONSTRUCTED_MIR9)
    ensure_directory(TABLES_DIR)
    
    if not os.path.exists(GROUND_TRUTH_FILE):
        raise Exception('No ground truth model file found!')
  
    ground_truth_matrix = adjacency_matrix_from_file(GROUND_TRUTH_FILE)

    results_table_path = os.path.join(TABLES_DIR, 'MIR9_evaluation.txt')

    with open(results_table_path, 'w') as results_table: 
    # if we don't have results_table file, python will create it
    # if we have results_table file, python will rewrite it from scratch
    
        # creating results_table header
        results_table.write(
                           'filename,datapoints,trajectory_len,mode,freq,hd_frac,shd_frac,prec,rec,f1\n'
                           )

        datasets = glob.glob(os.path.join(DATASETS_DIR, '*.txt'))

        for i, dataset_path in enumerate(datasets):
            dataset_name = os.path.basename(dataset_path)
            reconstructed_dataset_path = os.path.join(RECONSTRUCTED_MIR9, dataset_name.replace('.txt','.sif'))

            # --- running bnf on generated datasets ---
            run_bnfinder(dataset_path, reconstructed_dataset_path)
          
            reconstructed_matrix = adjacency_matrix_from_sif(reconstructed_dataset_path, GROUND_TRUTH_FILE)

            # --- calculating metrics ---
            hd_f, shd_f, p, r, f1 = compute_metrics(ground_truth_matrix,  reconstructed_matrix) 
            

            # --- filling results_table ---
            # finding dataset initial parameters from the name
            # example of dataset name : ds_NumPoints5_Len10_Async_Freq1.txt
            parts = dataset_name.split('_')
            num_points = re.search(r'\d+', parts[1]).group()
            traj_len = re.search(r'\d+', parts[2]).group()
            mode = parts[3]
            freq = re.search(r'\d+', parts[4]).group()

            results_table.write(
                                '{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'
                                .format(dataset_name, num_points, traj_len, mode, freq, hd_f, shd_f, p, r, f1)
                                )
 
            #progress
            if (i+1) % 5 == 0:
                print("Processed datasets = {}/{}".format(i+1, len(datasets)))

    print('----SUCCESS: MIR9 evaluation is ready at ', results_table_path)

if __name__ == "__main__":
    main()
