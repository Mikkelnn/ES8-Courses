from ai_handler import AiHandler
from logger import get_logger
from pathlib import Path
from model import *
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import keras.losses as kl
import keras.optimizers as ko
from tqdm import tqdm
import sklearn.metrics as sklearn
import shutil
from loaddata import Data

data = Data()

# GENEREL_PATH = Path("../../../")
GENEREL_PATH = Path(".")  # /scratch # Use full path for correct mapping on ai-lab container
RESULTS_PATH = GENEREL_PATH / "results"
TRAINING_DATA_PATH = GENEREL_PATH / "zero_one/training_data" # "big_training_data"
VALIDATE_DATA_PATH = GENEREL_PATH / "zero_one/validate_data" # "training_data"


log = get_logger()
ai_handler = AiHandler(RESULTS_PATH) # namedResultDir="12-12-2025_09:51:09"



def main():
    log.info(f"PYTHON_NUM_THREADS: {mp.cpu_count()}")
    log.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    log.info(f"TF_NUM_INTRAOP_THREADS: {os.environ.get('TF_NUM_INTRAOP_THREADS')}")
    log.info(f"TF_NUM_INTEROP_THREADS: {os.environ.get('TF_NUM_INTEROP_THREADS')}")

    with ai_handler.strategy.scope():
        model = None
        time_started = 0
        batch_size = 1 # Decrease as model get larger to fit in GPU memory
        epochs = 3
        initial_epoch = 0
        train_on_latest_result = False

        try:
            time_started = ai_handler.set_time_start()

            if train_on_latest_result:
                (found, initial_epoch, model) = ai_handler.find_latest_model()
                epochs += initial_epoch
                if not found:
                    exit()
            else:
                model = defineModel_image_10_classes()
            
            model.summary()

            # exit()

            ai_handler.plot_block_diagram(model)

            loss = kl.CategoricalFocalCrossentropy(
                gamma=2.0,
                alpha=0.25
            )

            compiled_model = ai_handler.compile_model(model,
                                    optimizer=ko.Adam(1e-4),
                                    loss=loss,
                                    metrics=["accuracy"]
                                    )

            # def loader_func_label(f): 
            #     label = np.load(f) # shape (2,) → [range, velocity]
            #     # return label
            #     return np.array([1,0]) if (sum(label) == 0) else np.array([0,1])
                
            # def loader_func_data(f): 
            #     # data = (np.load(f)[0])[... , None]
            #     data = np.load(f)
            #     return np.nan_to_num(data, nan=0.0)

            # labeld_data = ai_handler.dataset_from_data_and_labels(
            #     data_dir=TRAINING_DATA_PATH / "input",
            #     label_dir=TRAINING_DATA_PATH / "labels",
            #     batch_size=batch_size,
            #     shuffle=True,
            #     loader_func_label=loader_func_label,
            #     loader_func_data=loader_func_data
            # )
            # labeld_validation = ai_handler.dataset_from_data_and_labels(
            #     data_dir=VALIDATE_DATA_PATH / "input",
            #     label_dir=VALIDATE_DATA_PATH / "labels",
            #     batch_size=batch_size,
            #     shuffle=False,
            #     loader_func_label=loader_func_label,
            #     loader_func_data=loader_func_data
            # )

            train = ai_handler.tf.data.Dataset.from_tensor_slices((data.x_train, data.y_train))
            val = ai_handler.tf.data.Dataset.from_tensor_slices((data.x_test, data.y_test))

            buffer_size = 50000
            batch_size = 64

            train = train.shuffle(buffer_size).batch(batch_size).prefetch(ai_handler.tf.data.AUTOTUNE)
            val = val.batch(batch_size).prefetch(ai_handler.tf.data.AUTOTUNE)

            history = compiled_model.fit(
                train,
                validation_data=val,
                epochs=10,
                initial_epoch=0
            )

            ai_handler.set_time_stop()

            ai_handler.save_model(compiled_model)

            acc = history.history["accuracy"]
            val_acc = history.history["val_accuracy"]
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            epochs = range(1, len(acc) + 1)

            for i in epochs:
                log.info(
                    f"Epoch {i}: loss {loss[i - 1]}, validation loss {val_loss[i - 1]}, accuracy {acc[i - 1]}, validation accuracy {val_acc[i - 1]}"
                )


            # Plot and save accuracy figure
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, acc, label="Training Accuracy")
            plt.plot(epochs, val_acc, label="Validation Accuracy")
            plt.title("Model Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(ai_handler.result_path / "accuracy.svg", format="svg")
            plt.savefig(ai_handler.result_path / "accuracy.png", format="png")
            plt.close()

            # Plot and save loss figure
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, loss, label="Training Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.title("Model Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(ai_handler.result_path / "loss.svg", format="svg")
            plt.savefig(ai_handler.result_path / "loss.png", format="png")
            plt.close()

            #confusion_matrix()

        except Exception as e:
            # pass
            log.error(f"An error occurred during model training: {e}")
        finally:
            # copy logfiles
            log.info("Copying log files to result directory")
            src_dir = Path(__file__).parent.parent
            for f in ["log.log", "my_job.err", "my_job.out"]:
                src = os.path.join(src_dir, f)
                if os.path.isfile(src):
                    shutil.copy2(src, ai_handler.result_path)


def load_predict(modelPath = "results/26-09-2025_12:15:53/sum_diff_model.keras"):
    model = ai_handler.load_model(modelPath)
    res = ai_handler.predict(model, [0.1, 0.3])
    print(res)

# def confusion_matrix():
#     modelPath = "../models/detector_15_12_2025"

#     model = ai_handler.load_model_directory(modelPath)
    
#     def loader_func_data(f): 
#         # data = np.load(f)[... , None]
#         data = np.load(f)
#         return np.nan_to_num(data, nan=0.0)

#     def loader_func_label(f):
#         arr = np.load(f)
#         # class 0: no debris (sum == 0), class 1: debris present (sum != 0)        
#         return 0 if (np.sum(arr) == 0) else 1
    
#         #return np.array([1,0]) if (sum(np.load(f)) == 0) else np.array([0,1])

#     data_dir, label_dir = Path(TRAINING_DATA_PATH / "input"), Path(TRAINING_DATA_PATH / "labels")
#     data_files  = sorted(str(f) for f in Path(data_dir).glob("*"))
#     label_files = sorted(str(f) for f in Path(label_dir).glob("*"))
#     assert len(data_files) == len(label_files), "Data and label counts differ"
    
#     # N, TP, FP, TN, FN = len(data_files), 0, 0, 0, 0

#     y_true = []
#     y_pred = []

#     for data_file, label_file in tqdm(zip(data_files, label_files), total=len(data_files)):
#         pre = ai_handler.predict(model, loader_func_data(data_file))

#         pre_idx = int(np.argmax(pre, axis=-1)) 
#         act_idx = loader_func_label(label_file) 

#         label = np.load(label_file)

#         log.info(f"pre: {pre}, label_raw: {label}, label_range: {label[0] * 1000} m, label_velocity: {label[1] * 7500} m/s")

#         y_true.append(act_idx)
#         y_pred.append(pre_idx)

#         #if np.array_equal(act, [0,1]) and np.array_equal(pre, [0,1]):
#         #    TP += 1
#         #elif np.array_equal(act, [0,1]) and np.array_equal(pre, [1,0]):
#         #    FN += 1
#         #elif np.array_equal(act, [1,0]) and np.array_equal(pre, [0,1]):
#         #    FP += 1
#         #elif np.array_equal(act, [1,0]) and np.array_equal(pre, [1,0]):
#         #    TN += 1

#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     log.info(f"count: {len(y_true)}, targets: {np.sum(y_true==1)}, no targets: {np.sum(y_true==0)}")

#     #TP /= (TP + FN)
#     #FN /= (TP + FN)
#     #FP /= (FP + TN)
#     #TN /= (FP + TN)
    
#     cm_counts = sklearn.confusion_matrix(y_true, y_pred, labels=[1, 0])

#     #cm = np.array([[TP, FN], [FP, TN]])
    
#     TP, FN = cm_counts[0]
#     FP, TN = cm_counts[1]
    
#     # Optionally normalize per row (true class)
#     cm_norm = cm_counts.astype(float)
#     cm_norm[0] /= (TP + FN) if (TP + FN) > 0 else 1.0  # positive class row
#     cm_norm[1] /= (FP + TN) if (FP + TN) > 0 else 1.0  # negative class row

#     # Plot normalized confusion matrix
#     plt.figure(figsize=(6, 6))
#     plt.imshow(cm_norm, cmap='viridis', vmin=0.0, vmax=1.0)

#     for i in range(cm_norm.shape[0]):
#         for j in range(cm_norm.shape[1]):
#             plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center', color='black', fontsize=16)

#     plt.xticks([0, 1], ['1', '0'])
#     plt.yticks([0, 1], ['1', '0'])
#     plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
#     plt.gca().xaxis.set_label_position('top')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.gca().spines[:].set_visible(False)

#     plt.savefig(ai_handler.result_path / "confusion_matrix.svg", format="svg")
#     plt.close()

#     return cm_counts, cm_norm

#     # Save
#     #plt.figure(figsize=(6, 6))
#     #plt.imshow(cm, cmap='viridis')

#     # Add numbers in the middle of tiles
#     #for i in range(cm.shape[0]):
#     #    for j in range(cm.shape[1]):
#     #        plt.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center', color='black', fontsize=16)

#     # Add axis actual
#     #plt.xticks([0, 1], ['1', '0'])
#     #plt.yticks([0, 1], ['1', '0'])
#     #plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
#     #plt.gca().xaxis.set_label_position('top')
#     #plt.xlabel('Predicted')
#     #plt.ylabel('Actual')
#     #plt.gca().spines[:].set_visible(False)

#     #plt.savefig(ai_handler.result_path / "confusion_matrix.svg", format="svg")
#     #plt.close()
    
#     #return cm

def confusion_matrix_cifar10(model, data, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # --- Get data ---
    x_test = data.x_test
    y_test = data.y_test
    class_labels = data.CLASS_LABELS

    # --- Convert one-hot → class index ---
    y_true = np.argmax(y_test, axis=1)

    # --- Predictions ---
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    # --- Confusion matrix (counts) ---
    cm_counts = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    # --- Normalize (row-wise) ---
    cm_norm = cm_counts.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    # --- Confidence intervals (Wilson score, 95%) ---
    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return 0.0, 0.0
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        margin = (z * np.sqrt((p*(1-p)/n) + (z**2/(4*n**2)))) / denom
        return center - margin, center + margin

    ci_low = np.zeros_like(cm_norm)
    ci_high = np.zeros_like(cm_norm)

    for i in range(10):
        n = row_sums[i, 0]
        for j in range(10):
            k = cm_counts[i, j]
            low, high = wilson_ci(k, n)
            ci_low[i, j] = low
            ci_high[i, j] = high

    # --- Plot ---
    plt.figure(figsize=(10, 10))
    plt.imshow(cm_norm, cmap='viridis', vmin=0.0, vmax=1.0)

    for i in range(10):
        for j in range(10):
            plt.text(
                j, i,
                f"{cm_norm[i,j]:.2f}\n[{ci_low[i,j]:.2f},{ci_high[i,j]:.2f}]",
                ha='center', va='center', fontsize=7
            )

    plt.xticks(range(10), class_labels, rotation=45)
    plt.yticks(range(10), class_labels)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Normalized with 95% CI)')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg")
    else:
        plt.show()

    plt.close()

    return cm_counts, cm_norm, ci_low, ci_high

if __name__ == "__main__":
    #load_predict()
    main()
    #_ = confusion_matrix()