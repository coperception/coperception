import os
import argparse
import pandas as pd
import numpy as np

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="SORT demo")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--cross", type=str, help='[with_cross / no_cross]')
    parser.add_argument("--scene_idxes_file", type=str, help="File containing idxes of scenes to run tracking")
    parser.add_argument(
        "--from_agent", default=0, type=int, help="start from which agent"
    )
    parser.add_argument(
        "--to_agent", default=6, type=int, help="until which agent (index + 1)"
    )
    parser.add_argument("--split", type=str, help="[test/val]")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    mode = args.mode
    cross = args.cross
    scene_idxes_file = args.scene_idxes_file
    from_agent = args.from_agent
    to_agent = args.to_agent
    split = args.split

    print(args)

    result_dict = {}
    for agent_idx in range(from_agent, to_agent):
        result_dict[agent_idx] = []
    
    # run eval for MOTA and MOTP
    for current_agent in range(from_agent, to_agent):
        os.system(f'python ./TrackEval/scripts/run_mot_challenge.py --BENCHMARK V2X --SPLIT_TO_EVAL {split}{current_agent} --TRACKERS_TO_EVAL sort-{mode}/{cross} --METRICS CLEAR --DO_PREPROC False')

        # collect results
        eval_output_path = f'./TrackEval/data/trackers/mot_challenge/V2X-test{current_agent}/sort-{mode}/{cross}/pedestrian_summary.txt'
        eval_output_file = open(eval_output_path, 'r')
        # skip header
        eval_output_file.readline()
        perfs = eval_output_file.readline().split(' ')
        
        # MOTA and MOTP
        result_dict[current_agent].append(float(perfs[0]))
        result_dict[current_agent].append(float(perfs[1]))

    # run eval for other metrics
    for current_agent in range(from_agent, to_agent):
        os.system(f'python ./TrackEval/scripts/run_mot_challenge.py --BENCHMARK V2X --SPLIT_TO_EVAL {split}{current_agent} --TRACKERS_TO_EVAL sort-{mode}/{cross} --METRICS HOTA --DO_PREPROC False')

        # collect results
        eval_output_path = f'./TrackEval/data/trackers/mot_challenge/V2X-test{current_agent}/sort-{mode}/{cross}/pedestrian_summary.txt'
        eval_output_file = open(eval_output_path, 'r')
        # skip header
        eval_output_file.readline()
        perfs = eval_output_file.readline().split(' ')
        
        # HOTA DetA AssA DetRe DetPr AssRe AssPr LocA
        for ii in range(8):
            result_dict[current_agent].append(float(perfs[ii]))
    
    all_rows = []
    for current_agent in range(from_agent, to_agent):
        all_rows.append([current_agent] + result_dict[current_agent])

    mat = np.array(all_rows)
    mean = mat.mean(axis=0)
    all_rows.append(['mean'] + list(mean)[1:])
    df = pd.DataFrame(all_rows, columns=['agent', 'MOTA', 'MOTP', 'HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA'])
    os.makedirs('logs', exist_ok=True)
    df.to_csv(f'logs/logs_{mode}_{cross}.csv', sep=',', index=False)
