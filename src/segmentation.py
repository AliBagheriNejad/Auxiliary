import pandas as pd
import numpy as np
import os
import pickle


def segment(df, ws, ol, is_array=False, ws_2=None):
    if is_array:
        signals = df
    else:
        signals = df.to_numpy()
    step_size = int(ws*(1-ol))
    if len(signals.shape)==2:
        windows = np.lib.stride_tricks.sliding_window_view(signals, (ws,2))[::step_size,:]
    elif len(signals.shape)==3:
        windows = np.lib.stride_tricks.sliding_window_view(signals, (ws,ws_2,2))[::step_size]
    windows = np.squeeze(windows)
    return windows

def save_array(array, name):

    with open(name, 'wb') as f:
        pickle.dump(array, f)

def main():

    def make_samples():
        df = pd.read_csv(f_path)
        windows = segment(df,WINDOW_SIZE,OVERLAP)
        new_windows = segment(windows,NUM_WINDWOS,0,True,WINDOW_SIZE)
        return new_windows
        

    WINDOW_SIZE = 1024
    OVERLAP = 0.5
    NUM_WINDWOS = 5

    data_dir = r'D:\Masters\CWRU-dataset\48k_Drive_End_Bearing_Fault_Data'

    total_x = None
    total_y = None
    for root, dirs, files in os.walk(data_dir):
        
        if len(files) != 0:

            files_csv = [f for f in files if f.endswith('csv') and (('_2.' in f) or ('_3.' in f))]
            for f in files_csv:

                f_path = os.path.join(root,f)
                label = f.split('_')[0]

                new_windows = make_samples()
                label_array = np.array([[label]*NUM_WINDWOS]*len(new_windows))#.reshape(-1,NUM_WINDWOS)

                if total_x is None:
                    total_x = new_windows
                    total_y = label_array
                else:
                    total_x = np.vstack([total_x,new_windows])
                    total_y = np.vstack([total_y,label_array])


    save_array(total_x, r'F:\thesis\Articles\2nd\code\Data\input.pkl')
    save_array(total_y, r'F:\thesis\Articles\2nd\code\Data\output.pkl')


    print(total_x.shape, total_y.shape)


if __name__=='__main__':
    main()

