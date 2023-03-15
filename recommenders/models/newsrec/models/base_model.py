# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.compat.v1 import keras

from recommenders.models.deeprec.deeprec_utils import cal_metric

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
__all__ = ["BaseModel"]


class BaseModel:
    """Basic class of models

    Attributes:
        hparams (HParams): A HParams object, holds the entire set of hyperparameters.
        train_iterator (object): An iterator to load the data in training steps.
        test_iterator (object): An iterator to load the data in testing steps.
        graph (object): An optional graph.
        seed (int): Random seed.
    """

    def __init__(
            self,
            hparams,
            iterator_creator,
            seed=None,
    ):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function,
        parameter set.

        Args:
            hparams (HParams): A HParams object, holds the entire set of hyperparameters.
            iterator_creator (object): An iterator to load the data.
            graph (object): An optional graph.
            seed (int): Random seed.
        """
        self.seed = seed
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

        self.train_iterator = iterator_creator(
            hparams,
            hparams.npratio,
            col_spliter="\t",
        )
        self.test_iterator = iterator_creator(
            hparams,
            col_spliter="\t",
        )

        self.hparams = hparams
        self.support_quick_scoring = hparams.support_quick_scoring

        # set GPU use with on demand growth
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        )

        # set this TensorFlow session as the default session for Keras
        tf.compat.v1.keras.backend.set_session(sess)

        # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
        # Otherwise, their weights will be unavailable in the threads after the session there has been set
        self.model, self.scorer = self._build_graph()

        self.loss = self._get_loss()
        self.train_optimizer = self._get_opt()

        self.model.compile(loss=self.loss, optimizer=self.train_optimizer)

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.

        Args:
            file_path (str): the pre-trained glove embeddings file path.

        Returns:
            numpy.ndarray: A constant numpy array.
        """

        return np.load(file_path)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    @abc.abstractmethod
    def _get_input_label_from_iter(self, batch_data):
        """Subclass will implement this"""
        pass

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss

        Returns:
            object: Loss function or loss function name
        """
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif self.hparams.loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _get_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adam":
            train_opt = keras.optimizers.Adam(lr=lr)

        return train_opt

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.

        Args:
            logit (object): Base prediction value.
            task (str): A task (values: regression/classification)

        Returns:
            object: Transformed score
        """
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def train(self, train_batch_data):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (object): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        train_input, train_label = self._get_input_label_from_iter(train_batch_data)
        rslt = self.model.train_on_batch(train_input, train_label)
        return rslt

    def eval(self, eval_batch_data):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (object): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value, predicted scores, and ground-truth labels.
        """
        eval_input, eval_label = self._get_input_label_from_iter(eval_batch_data)
        imp_index = eval_batch_data["impression_index_batch"]

        pred_rslt = self.scorer.predict_on_batch(eval_input)

        return pred_rslt, eval_label, imp_index

    def fit(
            self,
            train_news_file,
            train_behaviors_file,
            valid_news_file,
            valid_behaviors_file,
            test_news_file=None,
            test_behaviors_file=None,
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_news_file is not None, evaluate it too.

        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_news_file (str): test set.

        Returns:
            object: An instance of self.
        """

        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            train_start = time.time()

            tqdm_util = tqdm(
                self.train_iterator.load_data_from_file(
                    train_news_file, train_behaviors_file
                )
            )

            for batch_data_input in tqdm_util:

                step_result = self.train(batch_data_input)
                step_data_loss = step_result

                epoch_loss += step_data_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    tqdm_util.set_description(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, epoch_loss / step, step_data_loss
                        )
                    )

            train_end = time.time()
            train_time = train_end - train_start

            eval_start = time.time()

            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", epoch_loss / step)]
                ]
            )

            eval_res = self.run_eval(valid_news_file, valid_behaviors_file)
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            if test_news_file is not None:
                test_res = self.run_eval(test_news_file, test_behaviors_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_news_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )
            )

        return self

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.

        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.

        Returns:
            list, list, list:
            - Keys after group.
            - Labels after group.
            - Preds after group.

        """

        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for label, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(label)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_keys, all_labels, all_preds

    def run_eval(self, news_filename, behaviors_file):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """

        if self.support_quick_scoring:
            _, group_labels, group_preds, group_preds_reldiff = self.run_fast_eval(
                news_filename, behaviors_file
            )
        else:
            _, group_labels, group_preds = self.run_slow_eval(
                news_filename, behaviors_file
            )
            group_preds_reldiff = group_preds  # error catch run_slow_eval not implemented for RelDiff
        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        # Added to print the metrics, the RelDiff ones are returned
        res_reldiff = cal_metric(group_labels, group_preds_reldiff, self.hparams.metrics)
        print("pred       ", res, "\npred_reldiff", res_reldiff)
        return res_reldiff

    def user(self, batch_user_input):
        user_input = self._get_user_feature_from_iter(batch_user_input)
        user_vec = self.userencoder.predict_on_batch(user_input)
        user_index = batch_user_input["impr_index_batch"]

        return user_index, user_vec

    def user_reldiff(self, batch_user_input):
        user_input = self._get_user_feature_from_iter(batch_user_input)
        user_vec = self.userencoder.predict_on_batch(user_input)
        user_index = batch_user_input["impr_index_batch"]

        # print(user_input.shape, user_vec.shape, user_index.shape)
        # print(user_input)
        # print(5 + "i")

        # Get the user click history as array of news embeddings
#         user_n_news_vec = {}
#         for i, uc in enumerate(user_index):
#             # Sanitise input by removing zero arrays (as users have not necessarily interacted with 50 news)
#             a = user_input[i]
#             a = a[~np.all(a == 0, axis=1)]  # remove zero arrays
#             # Compatibility - add one zero vector back - the 'new user gets average news recommended solution'
#             if a.size == 0:
#                 a = np.array([np.zeros(user_input[i].shape[1])])

#             # TODO make the max number of news_history (n) be able to change easily
#             # Get the number of documents from user click history to return back, set limit to the number of amount of
#             #  clicked in the user history (whichever is lower)
#             n = min(24, a.shape[0])
#             # Create the dictionary required for self.news() and run it
# #             batch_user_news_input = {"news_index_batch": np.arange(0, n, 1),
# #                                      "candidate_title_batch": a}
# #             user_n_news_vec[uc] = self.news(batch_user_news_input)[1]
#             # user_n_news_vec[uc] = np.float32(self.newsencoder.predict_on_batch(a[:n, :]))
#             user_n_news_vec[uc] = user_input
        return user_index, user_vec, user_input

    def news(self, batch_news_input):
        news_input = self._get_news_feature_from_iter(batch_news_input)
        news_vec = self.newsencoder.predict_on_batch(news_input)
        news_index = batch_news_input["news_index_batch"]

        return news_index, news_vec

    # Modified method run_user that also returns the user_history used for the RelDiff calculation
    def run_user_reldiff(self, news_filename, behaviors_file):
        if not hasattr(self, "userencoder"):
            raise ValueError("model must have attribute userencoder")

        user_indexes = []
        user_vecs = []
        user_histories = []
        for batch_data_input in tqdm(
                self.test_iterator.load_user_from_file(news_filename, behaviors_file),
                desc="load_user_from_file"
        ):
            user_index, user_vec, user_input = self.user_reldiff(batch_data_input)
            user_indexes.extend(np.reshape(user_index, -1))
            user_vecs.extend(user_vec)
            # Include the user news click history
            user_histories.extend(batch_data_input["user_history_batch"])
            # print(batch_data_input)
            # print(np.array(user_indexes).shape)                   # (32, )
            # print(np.array(user_vecs).shape)                      # (32, 400)
            # print(np.array(user_histories).shape)                 # (32, 50)
            # print(crashed)
        print('user_indexes length:        ', np.array(user_indexes).shape)
        print('user_vecs length:           ', np.array(user_vecs).shape)
        print('user_n_news_vecs_all length:', np.array(user_histories).shape)
        return dict(zip(user_indexes, user_vecs)), dict(zip(user_indexes, user_histories))

    def run_user(self, news_filename, behaviors_file):
        if not hasattr(self, "userencoder"):
            raise ValueError("model must have attribute userencoder")

        user_indexes = []
        user_vecs = []
        for batch_data_input in tqdm(
                self.test_iterator.load_user_from_file(news_filename, behaviors_file)
        ):
            user_index, user_vec = self.user(batch_data_input)
            user_indexes.extend(np.reshape(user_index, -1))
            user_vecs.extend(user_vec)

        return dict(zip(user_indexes, user_vecs))

    def run_news(self, news_filename):
        if not hasattr(self, "newsencoder"):
            raise ValueError("model must have attribute newsencoder")

        news_indexes = []
        news_vecs = []
        for batch_data_input in tqdm(
                self.test_iterator.load_news_from_file(news_filename),
                desc="load_news_from_file"
        ):
            news_index, news_vec = self.news(batch_data_input)
            news_indexes.extend(np.reshape(news_index, -1))
            news_vecs.extend(news_vec)

        return dict(zip(news_indexes, news_vecs))

    def run_slow_eval(self, news_filename, behaviors_file):
        preds = []
        labels = []
        imp_indexes = []

        for batch_data_input in tqdm(
                self.test_iterator.load_data_from_file(news_filename, behaviors_file)
        ):
            step_pred, step_labels, step_imp_index = self.eval(batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
            imp_indexes.extend(np.reshape(step_imp_index, -1))

        group_impr_indexes, group_labels, group_preds = self.group_labels(
            labels, preds, imp_indexes
        )
        return group_impr_indexes, group_labels, group_preds
            
    def pr_matrix(self, m): # m = user_vec to numpy array (all of them) (50, 400)
        from scipy.linalg import orth
        # print(m.shape)
        # (10, 400)
        # maybe np.dot instead of @ ???
        A = orth(m.T)
        # A = np.dot(U, U.T)
        Pr = A @ np.linalg.inv(A.T @ A) @ A.T
        print(Pr.shape, A.shape)
        # (10, 10) (10,) (400, 400) (10, 10)
        return Pr # (10, 10) - expected (10, 400) or (400, 10) (400, 400) --> dot product with cn

    def reldiff(self, user, user_history, candidate_news):
        # TODO test Projection matrix
        rd = []
        for n in candidate_news:
            cn = n * user_history 

            """ L2 norm of each vector """
            l2 = np.linalg.norm(cn, axis=1)
            l2 = np.where(l2 == 0, 1, l2)
            rd.append(user - (cn.T / l2).T)

            # """ L2 norm of each vector with Projection Matrix """
            # l2 = np.linalg.norm(cn, axis=1)
            # l2 = np.where(l2 == 0, 1, l2)
            # rd.append(user - (self.prm @ (cn.T / l2)).T) # comment not to use Projection Matrix
            
            # """ without any normalisation """
            # rd.append(user - cn)

            # """ L2 norm of the whole matrix """
            # l2 = np.linalg.norm(cn)
            # l2 = np.where(l2 == 0, 1, l2)
            # rd.append(user - (cn / l2))

        # return np.mean(rd, axis=1)  # np.array of length len(candidate_news) that is the RelDiffs of user embeddings
        return rd
    
    def run_fast_eval(self, news_filename, behaviors_file, update=False, n=None):
        if update:
            self.test_iterator.update_datasets(news_filename, behaviors_file)
        news_vecs = self.run_news(news_filename)
        # Run the extended method that also saves embeddings of the user history
        # user_vecs = self.run_user(news_filename, behaviors_file)
        user_vecs, user_clicked_news = self.run_user_reldiff(news_filename, behaviors_file)

        self.prm = self.pr_matrix(np.stack(list(user_vecs.values())))

        self.news_vecs = news_vecs
        self.user_vecs = user_vecs

        group_impr_indexes = []
        group_labels = []
        group_preds = []
        group_preds_reldiff = []

        for (
                impr_index,
                news_index,
                user_index,
                label,
        ) in tqdm(self.test_iterator.load_impression_from_file(behaviors_file), desc="load_impression_from_file"):
            news_stack = np.stack([news_vecs[i] for i in news_index], axis=0)

            # pred = np.dot(
            #     news_stack,
            #     user_vecs[impr_index],
            # )

            # TODO get vectors from this
            user_history = user_clicked_news[impr_index]
            user_history = user_history[np.nonzero(user_history)]
            if n:
                user_history = user_history[:min(n, len(user_history))]
                ## user_history = user_history[len(user_history) - min(n, len(user_history)): ]
            if len(user_history) == 0: user_history = [0]
            # print(user_history)
            # print(type(user_history))
            # print(user_history.shape)           # (50, 30)
            # print(user_vecs[user_index].shape)  # (400, )
            # print(impr_index, user_index)       # 0 200461
            user_history = np.stack([news_vecs[i] for i in list(user_history)])


            # Call the reldiff helper function to obtain the "stack" after the RelDiff has been applied
            user_vecs_reldiff = self.reldiff(user_vecs[impr_index], user_history, news_stack)
            user_vecs_reldiff_a_lot_more_information = np.array(user_vecs_reldiff)
            user_vecs_reldiff = np.mean(user_vecs_reldiff, axis=1)

            # Calculate a dot product between the RelDiff embeddings and the normalised candidate_news==stack
            pred_reldiff = [np.dot(news, user) for news, user in zip(news_stack, user_vecs_reldiff)]

            group_impr_indexes.append(impr_index)
            group_labels.append(label)
            # group_preds.append(pred)
            # Modify to append and return the original predictions and also the RelDiff ones
            group_preds_reldiff.append(pred_reldiff)


            test_impr = [2, 5, 7, 10, 15]
            if impr_index in test_impr:
                import os
                import json
                # with open(os.path.join("/scratch/2483099d/lvl4/recommendersUofG/examples/00_quick_start", "nrms_rd_embds.txt"), 'w') as f:
                #     f.write('~'.join(str(user_history)) + '\n')
                #     f.write('~'.join(str(user_vecs_reldiff)) + '\n')
                #     f.write('~'.join(str(candidate_news)) + '\n')
                #     f.write('~'.join(str(user_vecs_reldiff_a_lot_more_information)) + '\n')
                with open(os.path.join(f"/scratch/2483099d/lvl4/recommendersUofG/examples/00_quick_start", "nrms_rd_embds-{impr_index}.json"), 'w') as f:
                    f.write(json.dumps(user_history.tolist()) + '\n')
                    f.write(json.dumps(user_vecs[impr_index].tolist()) + '\n')
                    f.write(json.dumps(user_vecs_reldiff.tolist()) + '\n')
                    f.write(json.dumps(news_stack.tolist()) + '\n')
                    f.write(json.dumps(user_vecs_reldiff_a_lot_more_information.tolist()) + '\n')
                    f.write(json.dumps((np.argsort(pred_reldiff)[::-1])).tolist() + '\n')
                    f.write(json.dumps((np.argsort(np.dot(news_stack, user_vecs[impr_index]))[::-1])).tolist() + '\n')
                if impr_index == test_impr[-1]:
                    return None, None, None, None

        group_preds = group_preds_reldiff
        return group_impr_indexes, group_labels, group_preds, group_preds_reldiff
