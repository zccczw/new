import argparse
import json
import datetime

from args import get_parser, str2bool
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor

if __name__ == "__main__":
    dataset = 'SMAP'
    num = '1'
    group = '1'
    model_id = '11102022_144653'
    load_scores = True
    save_output = True

    if dataset == "SMD":
        model_path = f"./output/{dataset}/{group}/{model_id}"
    elif dataset == "ASD":
        model_path = f"./output/{dataset}/{num}/{model_id}"
    elif dataset in ['MSL', 'SMAP', 'SMDALL']:
        model_path = f"./output/{dataset}/{model_id}"
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()
    model_args_path = f"{model_path}/config.txt"

    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)
    window_size = model_args.lookback
    print(model_args)

    dataset = model_args.dataset
    if model_id is None:
        if dataset == 'SMD':
            dir_path = f"./output/{dataset}/{group}"
        else:
            dir_path = f"./output/{dataset}"
        dir_content = os.listdir(dir_path)
        subfolders = [subf for subf in dir_content if os.path.isdir(f"{dir_path}/{subf}") and subf != "logs"]
        date_times = [datetime.datetime.strptime(subf, '%d%m%Y_%H%M%S') for subf in subfolders]
        date_times.sort()
        model_datetime = date_times[-1]
        model_id = model_datetime.strftime('%d%m%Y_%H%M%S')

    else:
        model_id = model_id

    # Check that model exist
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # Get configs of model
    print(f'Using model from {model_path}')

    # 检查模型是否在指定的数据集上训练
    if dataset.lower() != model_args.dataset.lower():
        raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {dataset}.")

    elif dataset == "SMD" and group != model_args.group:
        print(f"Model trained on SMD group {model_args.group}, but asked to predict SMD group {group}.")

    window_size = model_args.lookback
    normalize = model_args.normalize
    n_epochs = model_args.epochs
    batch_size = model_args.bs
    init_lr = model_args.init_lr
    val_split = model_args.val_split
    shuffle_dataset = model_args.shuffle_dataset
    use_cuda = model_args.use_cuda
    print_every = model_args.print_every
    group_index = model_args.group[0]
    index = model_args.group[2:]
    args_summary = str(model_args.__dict__)

    if dataset == 'SMD':
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset == 'ASD':
        (x_train, _), (x_test, y_test) = get_data(f"omi-{num}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP', 'SMDALL']:
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    target_dims = get_target_dims(model_args.dataset)
    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        gru_n_layers=model_args.gru_n_layers,
        gru_hid_dim=model_args.gru_hid_dim,
        forecast_n_layers=model_args.fc_n_layers,
        forecast_hid_dim=model_args.fc_hid_dim,
        dropout=model_args.dropout,
        k=model_args.topK,
        embed_size=model_args.embed_size,
        in_channels=model_args.in_channels,
        graph = model_args.graph
    )

    device = "cuda" if model_args.use_cuda and torch.cuda.is_available() else "cpu"
    load(model, f"{model_path}/model.pt", device=device)
    model.cuda()

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "ASD": (0.975, 0.01),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "SMDALL": (0.9960, 0.001)
    }
    key = "SMD-" + model_args.group[0] if model_args.dataset == "SMD" else model_args.dataset
    level, q = level_q_dict[key]
    if model_args.level is not None:
        level = model_args.level
    if model_args.q is not None:
        q = model_args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "ASD": 1, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "SMDALL": 1}
    key = "SMD-" + model_args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': model_args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': model_args.dynamic_pot,
        "use_mov_av": model_args.use_mov_av,
        "gamma": model_args.gamma,
        "reg_level": reg_level,
        "save_path": f"{model_path}",
    }

    # Creating a new summary-file each time when new prediction are made with a pre-trained model
    count = 0
    for filename in os.listdir(model_path):
        if filename.startswith("summary"):
            count += 1
    if count == 0:
        summary_file_name = "summary.txt"
    else:
        summary_file_name = f"summary_{count}.txt"

    # test process
    label = y_test[window_size:] if y_test is not None else None
    predictor = Predictor(model, window_size, n_features, prediction_args, summary_file_name=summary_file_name)
    predictor.predict_anomalies(x_train, x_test, label,
                                load_scores=load_scores,
                                save_output=save_output)
