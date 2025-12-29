import argparse
import json
import numpy as np
from scipy import linalg
from utils.load_utils import *
# import pdb

def l2_distance(ref, pred):
    # extracting facial features (exp and jaw_pose)
    ref_face = np.concatenate((ref[:, :, :50], ref[:, :, 176:179]), axis=2)
    pred_face = np.concatenate((pred[:, :, :50], pred[:, :, 176:179]), axis=2)
    
    # extract body features (excluding exp and jaw_pose)
    ref_body = np.concatenate((ref[:, :, 50:176], ref[:, :, 179:]), axis=2)
    pred_body = np.concatenate((pred[:, :, 50:176], pred[:, :, 179:]), axis=2)
    
    # # Standardize (z-score normalization) for face features
    # mean_ref_face = np.mean(ref_face[:, :, :], axis=(0, 1), keepdims=True)
    # std_ref_face = np.std(ref_face[:, :, :], axis=(0, 1), keepdims=True) + 1e-10
    # ref_face = (ref_face[:, :, :] - mean_ref_face) / std_ref_face
    # pred_face = (pred_face[:, :, :] - mean_ref_face) / std_ref_face
    
    # # Standardize (z-score normalization) for body features
    # mean_ref_body = np.mean(ref_body[:, :, :], axis=(0, 1), keepdims=True)
    # std_ref_body = np.std(ref_body[:, :, :], axis=(0, 1), keepdims=True) + 1e-10
    # ref_body = (ref_body[:, :, :] - mean_ref_body) / std_ref_body
    # pred_body = (pred_body[:, :,:] - mean_ref_body) / std_ref_body
    
    
    # calculate L2 distance for face features
    face_l2 = np.mean(np.linalg.norm(pred_face - ref_face, axis=-1))
    
    # calculate L2 distance for body features
    body_l2 = np.mean(np.linalg.norm(pred_body - ref_body, axis=-1))
    
    return face_l2, body_l2

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    Code apapted from https://github.com/mseitzer/pytorch-fid

    Copyright 2018 Institute of Bioinformatics, JKU Linz
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    mu and sigma are calculated through:
    ```
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    ```
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_frechet_feature_distance(ref, pred):
    # extracting facial features (exp and jaw_pose)
    ref_face = np.concatenate((ref[:, :, :50], ref[:, :, 176:179]), axis=2)
    pred_face = np.concatenate((pred[:, :, :50], pred[:, :, 176:179]), axis=2)
    
    # extract body features (excluding exp and jaw_pose)
    ref_body = np.concatenate((ref[:, :, 50:176], ref[:, :, 179:]), axis=2)
    pred_body = np.concatenate((pred[:, :, 50:176], pred[:, :, 179:]), axis=2)
    
    # reshape features
    ref_face = ref_face.reshape(-1, ref_face.shape[2])
    pred_face = pred_face.reshape(-1, pred_face.shape[2])
    ref_body = ref_body.reshape(-1, ref_body.shape[2])
    pred_body = pred_body.reshape(-1, pred_body.shape[2])

    # calculate mean and covariance for face features
    mean_ref_face = np.mean(ref_face, axis=0)
    mean_pred_face = np.mean(pred_face, axis=0)
    cov_ref_face = np.cov(ref_face, rowvar=False)
    cov_pred_face = np.cov(pred_face, rowvar=False)

    # calculate mean and covariance for body features
    mean_ref_body = np.mean(ref_body, axis=0)
    mean_pred_body = np.mean(pred_body, axis=0)
    cov_ref_body = np.cov(ref_body, rowvar=False)
    cov_pred_body = np.cov(pred_body, rowvar=False)

    # normalization for face features
    std_face = np.std(ref_face, axis=0) + 1e-10
    ref_face = (ref_face - mean_ref_face) / std_face
    pred_face = (pred_face - mean_ref_face) / std_face

    # normalization for body features
    std_body = np.std(ref_body, axis=0) + 1e-10
    ref_body = (ref_body - mean_ref_body) / std_body
    pred_body = (pred_body - mean_ref_body) / std_body

    # calculate Frechet distance for face features
    fdist_face = calculate_frechet_distance(
        mu1=np.mean(ref_face, axis=0), 
        sigma1=np.cov(ref_face, rowvar=False),
        mu2=np.mean(pred_face, axis=0), 
        sigma2=np.cov(pred_face, rowvar=False),
    )

    # calculate Frechet distance for body features
    fdist_body = calculate_frechet_distance(
        mu1=np.mean(ref_body, axis=0), 
        sigma1=np.cov(ref_body, rowvar=False),
        mu2=np.mean(pred_body, axis=0), 
        sigma2=np.cov(pred_body, rowvar=False),
    )
    
    return fdist_face, fdist_body

def paired_frechet_feature_distance(ref, pred, speak):
    # extracting concatenated facial features (exp and jaw_pose)
    ref_face = np.concatenate((ref[:, :, :50], ref[:, :, 176:179], speak[:, :, :50], speak[:, :, 176:179]), axis=-1)
    pred_face = np.concatenate((pred[:, :, :50], pred[:, :, 176:179], speak[:, :, :50], speak[:, :, 176:179]), axis=-1)
    
    # extract concatenated body features (excluding exp and jaw_pose)
    ref_body = np.concatenate((ref[:, :, 50:176], ref[:, :, 179:], speak[:, :, 50:176], speak[:, :, 179:]), axis=-1)
    pred_body = np.concatenate((pred[:, :, 50:176], pred[:, :, 179:], speak[:, :, 50:176], speak[:, :, 179:]), axis=-1)
    
    # reshape features
    ref_face = ref_face.reshape(-1, ref_face.shape[2])
    pred_face = pred_face.reshape(-1, pred_face.shape[2])
    ref_body = ref_body.reshape(-1, ref_body.shape[2])
    pred_body = pred_body.reshape(-1, pred_body.shape[2])

    # calculate mean and covariance for face features
    mean_ref_face = np.mean(ref_face, axis=0)
    mean_pred_face = np.mean(pred_face, axis=0)
    cov_ref_face = np.cov(ref_face, rowvar=False)
    cov_pred_face = np.cov(pred_face, rowvar=False)

    # calculate mean and covariance for body features
    mean_ref_body = np.mean(ref_body, axis=0)
    mean_pred_body = np.mean(pred_body, axis=0)
    cov_ref_body = np.cov(ref_body, rowvar=False)
    cov_pred_body = np.cov(pred_body, rowvar=False)

    # normalization for face features
    std_face = np.std(ref_face, axis=0) + 1e-10
    ref_face = (ref_face - mean_ref_face) / std_face
    pred_face = (pred_face - mean_ref_face) / std_face

    # normalization for body features
    std_body = np.std(ref_body, axis=0) + 1e-10
    ref_body = (ref_body - mean_ref_body) / std_body
    pred_body = (pred_body - mean_ref_body) / std_body

    # calculate Frechet distance for face features
    fdist_face = calculate_frechet_distance(
        mu1=np.mean(ref_face, axis=0), 
        sigma1=np.cov(ref_face, rowvar=False),
        mu2=np.mean(pred_face, axis=0), 
        sigma2=np.cov(pred_face, rowvar=False),
    )

    # calculate Frechet distance for body features
    fdist_body = calculate_frechet_distance(
        mu1=np.mean(ref_body, axis=0), 
        sigma1=np.cov(ref_body, rowvar=False),
        mu2=np.mean(pred_body, axis=0), 
        sigma2=np.cov(pred_body, rowvar=False),
    )
    
    return fdist_face, fdist_body

def calculate_temporal_diversity(predictions):
    """
    Calculate the diversity score of model predictions based on variance along the temporal dimension.

    Args:
    predictions (numpy.ndarray): Model predictions with shape [B, T, F]
                                  where B is the batch size, T is the number of timestamps, 
                                  and F is the number of features.

    Returns:
    float: The diversity score representing the overall temporal diversity of the model's predictions.
    """
    # Calculate temporal variance for face_features (exp and jaw_pose)
    exp_features = predictions[:, :, :50]  # Extracting exp features (first 50 features)
    jaw_pose_features = predictions[:, :, 176:179]  # Extracting jaw_pose features (indices 176 to 178)
    face_features = np.concatenate((exp_features, jaw_pose_features), axis=2)
    temporal_variance_face = np.var(face_features, axis=1)


    # Calculate temporal variance for body_features (excluding exp and jaw_pose)
    body_features_part1 = predictions[:, :, 50:176]  # Features between exp and jaw_pose
    body_features_part2 = predictions[:, :, 179:]  # Features after jaw_pose
    body_features = np.concatenate((body_features_part1, body_features_part2), axis=2)
    temporal_variance_body = np.var(body_features, axis=1)


    # Average the variance over both batches (B) and features (F)
    diversity_score_face = np.mean(temporal_variance_face)
    diversity_score_body = np.mean(temporal_variance_body)

    return diversity_score_face, diversity_score_body


def main(args):


    with open(args.config) as f:
        config = json.load(f)
    pipeline = config['pipeline']
    tag = config['tag']

    ## setup VQ-VAE model
    with open(config['l_vqconfig']) as f:
        l_vqconfig = json.load(f)

    vq_configs = {'l_vqconfig': l_vqconfig, 's_vqconfig': None}

    ## load reference data or pixie Input predictions
    out_num = 1
    test_X, test_Y, test_audio, test_meta, _ = \
        load_test_data(config, pipeline, tag, out_num=out_num,
                       vqconfigs=vq_configs, smooth=True,
                       speaker=args.speaker,eval_mode = 'u', num_out=None)

    B, T, F = test_Y.shape[0], test_Y.shape[1], test_Y.shape[2]
    test_Y = test_Y.reshape(B * 2, T // 2, F)
    test_X = test_X.reshape(B * 2, T // 2, F)
    test_Y = test_Y[1:, : , :]
    test_X = test_X[:, : , :]
    
    pred = np.load('outputs/all_udiva/conditioned/conditioned_model_actual_pred_all_speaker.npy')
    ref = test_Y[:pred.shape[0], :, :]
    speak = test_X[:pred.shape[0], :, :]

    # pdb.set_trace()

    # L2 Distance
    face_l2, body_l2 = l2_distance(ref, pred)
    print("L2 Distance - face: ", round(face_l2, 3))
    print("L2 Distance - Body: ", round(body_l2, 3))

    # Frechet Feature Distance
    fdist_face, fdist_body = calculate_frechet_feature_distance(ref, pred)
    print("Frechet Distance (FD) - face: ", round(fdist_face, 3))
    print("Frechet Distance (FD)- Body: ", round(fdist_body, 3))

    # Paired Frechet Feature Distance with Speaker Features
    pfdist_face, pfdist_body = paired_frechet_feature_distance(ref, pred, speak)
    print("Paired Frechet Distance (P-FD) with Listener & Speaker - face: ", round(pfdist_face, 3))
    print("Paired Frechet Distance (P-FD) with Listener & Speaker - Body: ", round(pfdist_body, 3))

    # Temporal Diversity
    diversity_score_face, diversity_score_body = calculate_temporal_diversity(pred)
    print("Temporal Diversity Score (Variance) - face:", round(diversity_score_face, 3))
    print("Temporal Diversity Score (Variance)- Body:", round(diversity_score_body, 3))

# Run the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required = True)
    parser.add_argument('--speaker', type=str, required = True)
    args = parser.parse_args()
    print(args)
    main(args)
