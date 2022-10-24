import argparse
import datetime
import numpy as np
import torch
import torch.utils.data
from data_utils import ModelNetDataset, ModelNetDataset_H5PY, ScanObjectNNDataset
from models import PointNet
from sklearn import metrics, svm
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from utils import copy_parameters


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("SVM classification")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size in training [default: 32]")
    parser.add_argument("--num_point", type=int, default=1024, help="Point Number [default: 1024]")
    parser.add_argument("--feature_transform", action="store_true", help="Using feature transform in pointnet")
    parser.add_argument("--model_path", type=str, required=True, help="model pre-trained")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help="kind of dataset such as scanobjectnn|modelnet40|scanobjectnn10",
    )
    parser.add_argument("--data_aug", action="store_true", help="Using data augmentation for training phase")
    # parameter of pointnet
    return parser.parse_args()


def train():
    args = parse_args()
    print(args)
    if args.dataset_type == "modelnet40":
        dataset = ModelNetDataset(
            root=args.dataset_path, npoints=args.num_point, split="train", data_augmentation=args.data_aug
        )
        test_dataset = ModelNetDataset(
            root=args.dataset_path, npoints=args.num_point, split="test", data_augmentation=False
        )
    elif args.dataset_type == "modelnet40h5py":
        dataset = ModelNetDataset_H5PY(
            filelist=args.dataset_path + "/train.txt", num_point=args.num_point, data_augmentation=args.data_aug
        )

        test_dataset = ModelNetDataset_H5PY(
            filelist=args.dataset_path + "/test.txt", num_point=args.num_point, data_augmentation=False
        )
    elif args.dataset_type == "scanobjectnn":
        dataset = ScanObjectNNDataset(
            root=args.dataset_path, npoints=args.num_point, split="train", data_augmentation=args.data_aug
        )

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path, split="test", npoints=args.num_point, data_augmentation=False
        )
    elif args.dataset_type == "scanobjectnnbg":
        dataset = ScanObjectNNDataset(
            root=args.dataset_path, npoints=args.num_point, split="train", data_augmentation=args.data_aug
        )

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path, split="test", npoints=args.num_point, data_augmentation=False
        )
    elif args.dataset_type == "scanobjectnn10":
        dataset = ScanObjectNNDataset(
            root=args.dataset_path,
            npoints=args.num_point,
            small_data=True,
            split="train",
            data_augmentation=args.data_aug,
        )

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path, split="test", npoints=args.num_point, data_augmentation=False
        )
    else:
        exit("wrong dataset type")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False
    )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    print(len(dataset), len(test_dataset))
    # num_classes = len(dataset.classes)
    num_classes = dataset.num_classes
    classifier = PointNet(
        input_channels=3, num_classes=num_classes, global_feature=True, feature_transform=args.feature_transform
    )
    if args.model_path != "":
        classifier = copy_parameters(classifier, torch.load(args.model_path))
    classifier.cuda()
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    with torch.no_grad():
        classifier.eval()
        for points, target in tqdm(dataloader, total=len(dataloader), smoothing=0.9):
            target = target
            points, target = points.cuda(), target.long().cuda()

            global_feature = classifier(points)

            X_train.append(global_feature.cpu().numpy())
            Y_train.append(target.cpu().numpy())

        for points, target in tqdm(testdataloader, total=len(testdataloader), smoothing=0.9):
            target = target
            points, target = points.cuda(), target.long().cuda()

            global_feature = classifier(points)

            X_test.append(global_feature.cpu().numpy())
            Y_test.append(target.cpu().numpy())

    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)

    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)

    print("Train size: %d, Test size: %d" % (len(X_train), len(X_test)))

    """ === Simple Trial === """
    linear_svm = svm.SVC(kernel="linear")
    linear_svm.fit(X_train, Y_train)
    Y_pred = linear_svm.predict(X_test)
    linear_acc = metrics.accuracy_score(Y_test, Y_pred)
    print("\n", "Simple Linear SVC accuracy:", linear_acc, "\n")
    rbf_svm = svm.SVC(kernel="rbf")
    rbf_svm.fit(X_train, Y_train)
    Y_pred = rbf_svm.predict(X_test)
    print("Simple RBF SVC accuracy:", metrics.accuracy_score(Y_test, Y_pred), "\n")

    """ === Grid Search for SVM with RBF Kernel === """

    print("Now we use Grid Search to opt the parameters for SVM RBF kernel")

    gamma_range = np.outer(np.logspace(-4, 1, 6), np.array([1, 5])).flatten()
    C_range = np.outer(np.logspace(-1, 3, 5), np.array([1, 5])).flatten()
    parameters = {"kernel": ["rbf"], "C": C_range, "gamma": gamma_range}

    svm_clsf = svm.SVC()
    grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=8, verbose=1)

    start_time = datetime.datetime.now()
    print("Start Param Searching at {}".format(str(start_time)))
    grid_clsf.fit(X_train, Y_train)
    print("Elapsed time, param searching {}".format(str(datetime.datetime.now() - start_time)))
    sorted(grid_clsf.cv_results_.keys())

    Y_pred = grid_clsf.best_estimator_.predict(X_test)
    print("\n\n")
    print("=" * 37)
    print("Best Params via Grid Search Cross Validation on Train Split is: ", grid_clsf.best_params_)
    print("Best Model's Accuracy on Test Dataset: {}".format(metrics.accuracy_score(Y_test, Y_pred)))
    print("\n", "Simple Linear SVC accuracy:", linear_acc, "\n")


if __name__ == "__main__":
    train()
