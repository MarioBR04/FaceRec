from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function
import argparse
import logging
import math
import os
import sys
import mxnet as mx
import mxnet.optimizer as optimizer
import numpy as np
np.bool = bool
import pandas
from sklearn.model_selection import train_test_split

from datap import FaceImageIter
from datap import FaceImageIterList

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import common.face_image as face_image
from common.noise_sgd import NoiseSGD

sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))

import fmobilenet
# import lfw
import eval.verification as verification

sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None


class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(AccMetric, self).__init__(
            'acc', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count += 1
        if args.loss_type >= 2 and args.loss_type <= 5 and args.margin_verbose > 0:
            if self.count % args.ctx_num == 0:
                mbatch = self.count // args.ctx_num
                if mbatch == 1 or mbatch % args.margin_verbose == 0:
                    a = 0.0
                    b = 0.0
                    if len(preds) >= 4:
                        a = preds[-2].asnumpy()[0]
                        b = preds[-1].asnumpy()[0]
                    elif len(preds) == 3:
                        a = preds[-1].asnumpy()[0]
                        b = a
                    print('[%d][MARGIN]%f,%f' % (mbatch, a, b))
        # loss = preds[2].asnumpy()[0]
        # if len(self.losses)==20:
        #  print('ce loss', sum(self.losses)/len(self.losses))
        #  self.losses = []
        # self.losses.append(loss)
        preds = [preds[1]]  # use softmax output
        for label, pred_label in zip(labels, preds):
            # print(pred_label)
            # print(label.shape, pred_label.shape)
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32').flatten()
            label = label.asnumpy()
            if label.ndim == 2:
                label = label[:, 0]
            label = label.astype('int32').flatten()
            # print(label)
            # print('label',label)
            # print('pred_label', pred_label)
            assert label.shape == pred_label.shape
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)


class LossValueMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValueMetric, self).__init__(
            'lossvalue', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []

    def update(self, labels, preds):
        loss = preds[-1].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1.0
        gt_label = preds[-2].asnumpy()
        # print(gt_label)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--data-dir', default='/home/mariobr04/PycharmProjects/MobileFaceNet/input/lfw-dataset',
                        help='')
    parser.add_argument('--prefix', default='models/model',
                        help='directory to save model.')
    parser.add_argument('--pretrained', default='models/model,98',
                        help='')
    parser.add_argument('--retrain', action='store_true', default=True,  # was false
                        help='true means continue training.')
    parser.add_argument('--ckpt', type=int, default=1, help='')  # default = 1
    parser.add_argument('--network', default='s20', help='')  # default='s20'
    parser.add_argument('--version-se', type=int, default=0, help='')
    parser.add_argument('--version-input', type=int, default=1, help='')
    parser.add_argument('--version-output', type=str, default='E', help='')
    parser.add_argument('--version-unit', type=int, default=3, help='')
    parser.add_argument('--end-epoch', type=int, default=50,
                        help='training epoch size.')
    parser.add_argument('--noise-sgd', type=float, default=0.0, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--wd', type=float, default=0.0005, help='')
    parser.add_argument('--mom', type=float, default=0.9,
                        help='')
    parser.add_argument('--emb-size', type=int, default=512,
                        help='')
    parser.add_argument('--per-batch-size', type=int, default=128,
                        help='')
    parser.add_argument('--margin-m', type=float, default=0.35,
                        help='')
    parser.add_argument('--margin-s', type=float, default=64.0,
                        help='')
    parser.add_argument('--easy-margin', type=int, default=0,
                        help='')
    parser.add_argument('--margin-verbose', type=int, default=0,
                        help='')
    parser.add_argument('--c2c-threshold', type=float, default=0.0,
                        help='')
    parser.add_argument('--c2c-mode', type=int, default=-10,
                        help='')
    parser.add_argument('--output-c2c', type=int, default=0,
                        help='')
    parser.add_argument('--margin', type=int, default=4,
                        help='')
    parser.add_argument('--beta', type=float, default=1000.,
                        help='')
    parser.add_argument('--beta-min', type=float, default=5.,
                        help='')
    parser.add_argument('--beta-freeze', type=int, default=0,
                        help='')
    parser.add_argument('--gamma', type=float, default=0.12,
                        help='')
    parser.add_argument('--power', type=float, default=1.0,
                        help='')
    parser.add_argument('--scale', type=float, default=0.9993,
                        help='')
    parser.add_argument('--center-alpha', type=float, default=0.5, help='')
    parser.add_argument('--center-scale', type=float, default=0.003, help='')
    parser.add_argument('--images-per-identity', type=int, default=0, help='')
    parser.add_argument('--triplet-bag-size', type=int, default=3600, help='')
    parser.add_argument('--triplet-alpha', type=float, default=0.3, help='')
    parser.add_argument('--triplet-max-ap', type=float, default=0.0, help='')
    parser.add_argument('--verbose', type=int, default=2000, help='')
    parser.add_argument('--loss-type', type=int, default=2,  # default 1
                        help='')
    parser.add_argument('--incay', type=float, default=0.0,
                        help='feature incay')
    parser.add_argument('--use-deformable', type=int, default=0,
                        help='')
    parser.add_argument('--rand-mirror', type=int, default=1,
                        help='')
    parser.add_argument('--cutoff', type=int, default=0, help='')
    parser.add_argument('--patch', type=str, default='0_0_96_112_0',
                        help='')
    parser.add_argument('--lr-steps', type=str, default='120000', help='')
    parser.add_argument('--max-steps', type=int, default=0, help='')
    parser.add_argument('--target', type=str, default='lfw,cfp_ff,cfp_fp,agedb_30', help='')
    args = parser.parse_args()
    return args


def create_data(dataset_path):
    lfw_allnames = pandas.read_csv("input/lfw-dataset/lfw_allnames.csv")
    matchpairsDevTest = pandas.read_csv("input/lfw-dataset/matchpairsDevTest.csv")
    matchpairsDevTrain = pandas.read_csv(".input/lfw-dataset/matchpairsDevTrain.csv")
    mismatchpairsDevTest = pandas.read_csv("input/lfw-dataset/mismatchpairsDevTest.csv")
    mismatchpairsDevTrain = pandas.read_csv("input/lfw-dataset/mismatchpairsDevTrain.csv")
    pairs = pandas.read_csv(".input/lfw-dataset/pairs.csv")
    # tidy pairs data:
    pairs = pairs.rename(columns={'name': 'name1', 'Unnamed: 3': 'name2'})
    matched_pairs = pairs[pairs["name2"] - pandas.isnull()].drop("name2", axis=1)
    mismatched_pairs = pairs[pairs["namez"].notnull()]
    people = pandas.read_csv("input/lfw-dataset/people.csv")
    # remove null values
    people = people[people.name.notnull()]
    peopleDevTest = pandas.read_csv("input/lfw-dataset/peopleDevTest.csv")
    peopleDevTrain = pandas.read_csv("input/lfw-dataset/peopleDevTrain.csv")
    image_paths = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]
    image_paths[' image_path'] = 1 + image_paths.groupby('name ').cumcount()
    image_paths[' image_path'] = image_paths.image_path.apply(lambda x: "{0:0>4}" - format(x))
    image_paths['image_path'] = image_paths.name + image_paths.name + image_paths.image_path + ". jpg"
    image_paths = image_paths.drop("images", 1)
    lfw_train, lfw_test = train_test_split(image_paths, test_size=0.2)
    lfw_train = lfw_train.resetindex().drop("index", 1)
    lfw_test = lfw_test.reset_index().drop("index", 1)


def get_symbol(args, arg_params, aux_params):
    data_shape = (args.image_channel, args.image_h, args.image_w)
    image_shape = ",".join([str(x) for x in data_shape])
    margin_symbols = []
    print('init mobilenet', args.num_layers)
    embedding = fmobilenet.get_symbol(args.emb_size,
                                      version_se=args.version_se, version_input=args.version_input,
                                      version_output=args.version_output, version_unit=args.version_unit)
    all_label = mx.symbol.Variable('softmax_label')
    if not args.output_c2c:
        gt_label = all_label
    else:
        gt_label = mx.symbol.slice_axis(all_label, axis=1, begin=0, end=1)
        gt_label = mx.symbol.reshape(gt_label, (args.per_batch_size,))
        c2c_label = mx.symbol.slice_axis(all_label, axis=1, begin=1, end=2)
        c2c_label = mx.symbol.reshape(c2c_label, (args.per_batch_size,))
    assert args.loss_type >= 0
    extra_loss = None
    if args.loss_type == 0:  # softmax
        _weight = mx.symbol.Variable('fc7_weight')
        _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
        fc7 = mx.sym.FullyConnected(data=embedding, weight=_weight, bias=_bias, num_hidden=args.num_classes, name='fc7')
    elif args.loss_type == 1:  # sphere
        _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes,
                              weight=_weight,
                              beta=args.beta, margin=args.margin, scale=args.scale,
                              beta_min=args.beta_min, verbose=1000, name='fc7')
    elif args.loss_type == 8:  # centerloss, TODO
        _weight = mx.symbol.Variable('fc7_weight')
        _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
        fc7 = mx.sym.FullyConnected(data=embedding, weight=_weight, bias=_bias, num_hidden=args.num_classes, name='fc7')
        print('center-loss', args.center_alpha, args.center_scale)
        extra_loss = mx.symbol.Custom(data=embedding, label=gt_label, name='center_loss', op_type='centerloss', \
                                      num_class=args.num_classes, alpha=args.center_alpha, scale=args.center_scale,
                                      batchsize=args.per_batch_size)
    elif args.loss_type == 2:
        s = args.margin_s
        m = args.margin_m
        _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        if s > 0.0:
            nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
            fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                        name='fc7')
            if m > 0.0:
                if args.margin_verbose > 0:
                    zy = mx.sym.pick(fc7, gt_label, axis=1)
                    cos_t = zy / s
                    margin_symbols.append(mx.symbol.mean(cos_t))

                s_m = s * m
                gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=s_m, off_value=0.0)
                fc7 = fc7 - gt_one_hot

                if args.margin_verbose > 0:
                    new_zy = mx.sym.pick(fc7, gt_label, axis=1)
                    new_cos_t = new_zy / s
                    margin_symbols.append(mx.symbol.mean(new_cos_t))
        else:
            fc7 = mx.sym.FullyConnected(data=embedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                        name='fc7')
            if m > 0.0:
                body = embedding * embedding
                body = mx.sym.sum_axis(body, axis=1, keepdims=True)
                body = mx.sym.sqrt(body)
                body = body * m
                gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
                body = mx.sym.broadcast_mul(gt_one_hot, body)
                fc7 = fc7 - body

    elif args.loss_type == 3:
        s = args.margin_s
        m = args.margin_m
        assert args.margin == 2 or args.margin == 4
        _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy / s
        if args.margin_verbose > 0:
            margin_symbols.append(mx.symbol.mean(cos_t))
        if m > 1.0:
            t = mx.sym.arccos(cos_t)
            t = t * m
            body = mx.sym.cos(t)
            new_zy = body * s
            if args.margin_verbose > 0:
                new_cos_t = new_zy / s
                margin_symbols.append(mx.symbol.mean(new_cos_t))
            diff = new_zy - zy
            diff = mx.sym.expand_dims(diff, 1)
            gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
            body = mx.sym.broadcast_mul(gt_one_hot, diff)
            fc7 = fc7 + body
    elif args.loss_type == 4:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        assert m >= 0.0
        assert m < (math.pi / 2)
        _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy / s
        if args.output_c2c == 0:
            cos_m = math.cos(m)
            sin_m = math.sin(m)
            mm = math.sin(math.pi - m) * m
            threshold = math.cos(math.pi - m)
            if args.margin_verbose > 0:
                margin_symbols.append(mx.symbol.mean(cos_t))
            if args.easy_margin:
                cond = mx.symbol.Activation(data=cos_t, act_type='relu')
            else:
                cond_v = cos_t - threshold
                cond = mx.symbol.Activation(data=cond_v, act_type='relu')
            body = cos_t * cos_t
            body = 1.0 - body
            sin_t = mx.sym.sqrt(body)
            new_zy = cos_t * cos_m
            b = sin_t * sin_m
            new_zy = new_zy - b
            new_zy = new_zy * s
            if args.easy_margin:
                zy_keep = zy
            else:
                zy_keep = zy - s * mm
            new_zy = mx.sym.where(cond, new_zy, zy_keep)
        else:
            cos_m = mx.sym.sqrt(c2c_label)
            sin_m = 1.0 - c2c_label
            sin_m = mx.sym.sqrt(sin_m)
            body = cos_t * cos_t
            body = 1.0 - body
            sin_t = mx.sym.sqrt(body)
            new_zy = cos_t * cos_m
            b = sin_t * sin_m
            new_zy = new_zy - b
            new_zy = new_zy * s

        if args.margin_verbose > 0:
            new_cos_t = new_zy / s
            margin_symbols.append(mx.symbol.mean(new_cos_t))
        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7 + body
    elif args.loss_type == 5:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        assert m >= 0.0
        assert m < (math.pi / 2)
        _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        cos_a = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                      name='fc7')
        theta_a = mx.sym.arccos(cos_a)
        gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=m, off_value=0.0)
        theta_a = theta_a + gt_one_hot
        fc7 = math.pi / 2 - theta_a
        fc7 = fc7 * s
    elif args.loss_type == 10:  # marginal loss
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        params = [1.2, 0.3, 1.0]
        n1 = mx.sym.expand_dims(nembedding, axis=1)  # N,1,C
        n2 = mx.sym.expand_dims(nembedding, axis=0)  # 1,N,C
        body = mx.sym.broadcast_sub(n1, n2)  # N,N,C
        body = body * body
        body = mx.sym.sum(body, axis=2)  # N,N
        # body = mx.sym.sqrt(body)
        body = body - params[0]
        mask = mx.sym.Variable('extra')
        body = body * mask
        body = body + params[1]
        # body = mx.sym.maximum(body, 0.0)
        body = mx.symbol.Activation(data=body, act_type='relu')
        body = mx.sym.sum(body)
        body = body / (args.per_batch_size * args.per_batch_size - args.per_batch_size)
        extra_loss = mx.symbol.MakeLoss(body, grad_scale=params[2])
    elif args.loss_type == 11:  # npair loss
        params = [0.9, 0.2]
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        nembedding = mx.sym.transpose(nembedding)
        nembedding = mx.symbol.reshape(nembedding, (args.emb_size, args.per_identities, args.images_per_identity))
        nembedding = mx.sym.transpose(nembedding, axes=(2, 1, 0))  # 2*id*512
        # nembedding = mx.symbol.reshape(nembedding, (args.emb_size, args.images_per_identity, args.per_identities))
        # nembedding = mx.sym.transpose(nembedding, axes=(1,2,0)) #2*id*512
        n1 = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=1)
        n2 = mx.symbol.slice_axis(nembedding, axis=0, begin=1, end=2)
        # n1 = []
        # n2 = []
        # for i in xrange(args.per_identities):
        #  _n1 = mx.symbol.slice_axis(nembedding, axis=0, begin=2*i, end=2*i+1)
        #  _n2 = mx.symbol.slice_axis(nembedding, axis=0, begin=2*i+1, end=2*i+2)
        #  n1.append(_n1)
        #  n2.append(_n2)
        # n1 = mx.sym.concat(*n1, dim=0)
        # n2 = mx.sym.concat(*n2, dim=0)
        # rembeddings = mx.symbol.reshape(nembedding, (args.images_per_identity, args.per_identities, 512))
        # n1 = mx.symbol.slice_axis(rembeddings, axis=0, begin=0, end=1)
        # n2 = mx.symbol.slice_axis(rembeddings, axis=0, begin=1, end=2)
        n1 = mx.symbol.reshape(n1, (args.per_identities, args.emb_size))
        n2 = mx.symbol.reshape(n2, (args.per_identities, args.emb_size))
        cosine_matrix = mx.symbol.dot(lhs=n1, rhs=n2, transpose_b=True)  # id*id, id=N of N-pair
        data_extra = mx.sym.Variable('extra')
        data_extra = mx.sym.slice_axis(data_extra, axis=0, begin=0, end=args.per_identities)
        mask = cosine_matrix * data_extra
        # body = mx.sym.mean(mask)
        fii = mx.sym.sum_axis(mask, axis=1)
        fij_fii = mx.sym.broadcast_sub(cosine_matrix, fii)
        fij_fii = mx.sym.exp(fij_fii)
        row = mx.sym.sum_axis(fij_fii, axis=1)
        row = mx.sym.log(row)
        body = mx.sym.mean(row)
        extra_loss = mx.sym.MakeLoss(body)
    elif args.loss_type == 12:  # triplet loss
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        anchor = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=args.per_batch_size // 3)
        positive = mx.symbol.slice_axis(nembedding, axis=0, begin=args.per_batch_size // 3,
                                        end=2 * args.per_batch_size // 3)
        negative = mx.symbol.slice_axis(nembedding, axis=0, begin=2 * args.per_batch_size // 3, end=args.per_batch_size)
        ap = anchor - positive
        an = anchor - negative
        ap = ap * ap
        an = an * an
        ap = mx.symbol.sum(ap, axis=1, keepdims=1)  # (T,1)
        an = mx.symbol.sum(an, axis=1, keepdims=1)  # (T,1)
        triplet_loss = mx.symbol.Activation(data=(ap - an + args.triplet_alpha), act_type='relu')
        triplet_loss = mx.symbol.mean(triplet_loss)
        # triplet_loss = mx.symbol.sum(triplet_loss)/(args.per_batch_size//3)
        extra_loss = mx.symbol.MakeLoss(triplet_loss)
    elif args.loss_type == 13:  # triplet loss with angular margin
        m = args.margin_m
        sin_m = math.sin(m)
        cos_m = math.cos(m)
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        anchor = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=args.per_batch_size // 3)
        positive = mx.symbol.slice_axis(nembedding, axis=0, begin=args.per_batch_size // 3,
                                        end=2 * args.per_batch_size // 3)
        negative = mx.symbol.slice_axis(nembedding, axis=0, begin=2 * args.per_batch_size // 3, end=args.per_batch_size)
        ap = anchor * positive
        an = anchor * negative
        ap = mx.symbol.sum(ap, axis=1, keepdims=1)  # (T,1)
        an = mx.symbol.sum(an, axis=1, keepdims=1)  # (T,1)

        ap = mx.symbol.arccos(ap)
        an = mx.symbol.arccos(an)
        triplet_loss = mx.symbol.Activation(data=(ap - an + args.margin_m), act_type='relu')

        # body = ap*ap
        # body = 1.0-body
        # body = mx.symbol.sqrt(body)
        # body = body*sin_m
        # ap = ap*cos_m
        # ap = ap-body
        # triplet_loss = mx.symbol.Activation(data = (an-ap), act_type='relu')

        triplet_loss = mx.symbol.mean(triplet_loss)
        extra_loss = mx.symbol.MakeLoss(triplet_loss)
    elif args.loss_type == 9:  # coco loss
        centroids = []
        for i in range(args.per_identities):
            xs = mx.symbol.slice_axis(embedding, axis=0, begin=i * args.images_per_identity,
                                      end=(i + 1) * args.images_per_identity)
            mean = mx.symbol.mean(xs, axis=0, keepdims=True)
            mean = mx.symbol.L2Normalization(mean, mode='instance')
            centroids.append(mean)
        centroids = mx.symbol.concat(*centroids, dim=0)
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * args.coco_scale
        fc7 = mx.symbol.dot(nembedding, centroids, transpose_b=True)  # (batchsize, per_identities)
        # extra_loss = mx.symbol.softmax_cross_entropy(fc7, gt_label, name='softmax_ce')/args.per_batch_size
        # extra_loss = mx.symbol.BlockGrad(extra_loss)
    else:
        # embedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*float(args.loss_type)
        embedding = embedding * 5
        _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance') * 2
        fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes,
                              weight=_weight,
                              beta=args.beta, margin=args.margin, scale=args.scale,
                              beta_min=args.beta_min, verbose=100, name='fc7')

        # fc7 = mx.sym.Custom(data=embedding, label=gt_label, weight=_weight, num_hidden=args.num_classes,
        #                       beta=args.beta, margin=args.margin, scale=args.scale,
        #                       op_type='ASoftmax', name='fc7')
    if args.loss_type <= 1 and args.incay > 0.0:
        params = [1.e-10]
        sel = mx.symbol.argmax(data=fc7, axis=1)
        sel = (sel == gt_label)
        norm = embedding * embedding
        norm = mx.symbol.sum(norm, axis=1)
        norm = norm + params[0]
        feature_incay = sel / norm
        feature_incay = mx.symbol.mean(feature_incay) * args.incay
        extra_loss = mx.symbol.MakeLoss(feature_incay)
    # out = softmax
    # l2_embedding = mx.symbol.L2Normalization(embedding)

    # ce = mx.symbol.softmax_cross_entropy(fc7, gt_label, name='softmax_ce')/args.per_batch_size
    # out = mx.symbol.Group([mx.symbol.BlockGrad(embedding), softmax, mx.symbol.BlockGrad(ce)])
    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = None
    if args.loss_type < 10:
        softmax = mx.symbol.SoftmaxOutput(data=fc7, label=gt_label, name='softmax', normalization='valid')
        out_list.append(softmax)
    if softmax is None:
        out_list.append(mx.sym.BlockGrad(gt_label))
    if extra_loss is not None:
        out_list.append(extra_loss)
    for _sym in margin_symbols:
        _sym = mx.sym.BlockGrad(_sym)
        out_list.append(_sym)
    out = mx.symbol.Group(out_list)
    return out, arg_params, aux_params


def train_net(args):
    highest_accT = 0.0  # Rastrea el mejor accuracy hasta el momento
    best_epoch = 0  # Época donde se alcanzó el mejor accuracy


    ctx = []
    cvd = '0'
    if len(cvd) > 0:
        for i in range(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx) == 0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx))
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    if args.per_batch_size == 0:
        args.per_batch_size = 128
        if args.loss_type == 10:
            args.per_batch_size = 256
    args.batch_size = args.per_batch_size * args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = 3
    ppatch = [int(x) for x in args.patch.split('_')]
    assert len(ppatch) == 5

    os.environ['BETA'] = str(args.beta)
    data_dir_list = args.data_dir.split(',')
    if args.loss_type != 12 and args.loss_type != 13:
        assert len(data_dir_list) == 1
    data_dir = data_dir_list[0]
    args.use_val = False
    path_imgrec = None
    path_imglist = None
    val_rec = None
    prop = face_image.load_property(data_dir)  # Load propertys
    args.num_classes = prop.num_classes
    image_size = prop.image_size
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)

    assert (args.num_classes > 0)
    print('num_classes', args.num_classes)
    args.coco_scale = 0.5 * math.log(float(args.num_classes - 1)) + 3

    path_imgrec = os.path.join(data_dir, "lfw_landmarks.rec")
    val_rec = os.path.join(data_dir, "val.rec")
    if os.path.exists(val_rec) and args.loss_type < 10:
        args.use_val = True
    else:
        val_rec = None
    # args.use_val = False
#Images per identity--------------------------------------------------------very important
    if args.loss_type == 1 and args.num_classes > 20000:
        args.beta_freeze = 5000
        args.gamma = 0.06

    if args.loss_type < 9:
        assert args.images_per_identity == 0
    else:
        if args.images_per_identity == 0:
            if args.loss_type == 11:
                args.images_per_identity = 2
            elif args.loss_type == 10 or args.loss_type == 9:
                args.images_per_identity = 16
            elif args.loss_type == 12 or args.loss_type == 13:
                args.images_per_identity = 5
                assert args.per_batch_size % 3 == 0
        assert args.images_per_identity >= 2
        args.per_identities = int(args.per_batch_size / args.images_per_identity)

    print('Called with argument:', args)

    data_shape = (args.image_channel, image_size[0], image_size[1])
    mean = None

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained) == 0:
        arg_params = None
        aux_params = None
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    else:
        vec = args.pretrained.split(',')
        print('loading', vec)
        mx.npx.reset_np()
        _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

    data_extra = None
    hard_mining = False
    triplet_params = None
    coco_mode = False
    if args.loss_type == 10:
        hard_mining = True
        _shape = (args.batch_size, args.per_batch_size)
        data_extra = np.full(_shape, -1.0, dtype=np.float32)
        c = 0
        while c < args.batch_size:
            a = 0
            while a < args.per_batch_size:
                b = a + args.images_per_identity
                data_extra[(c + a):(c + b), a:b] = 1.0
                # print(c+a, c+b, a, b)
                a = b
            c += args.per_batch_size
    elif args.loss_type == 11:
        data_extra = np.zeros((args.batch_size, args.per_identities), dtype=np.float32)
        c = 0
        while c < args.batch_size:
            for i in range(args.per_identities):
                data_extra[c + i][i] = 1.0
            c += args.per_batch_size
    elif args.loss_type == 12 or args.loss_type == 13:
        triplet_params = [args.triplet_bag_size, args.triplet_alpha, args.triplet_max_ap]
    elif args.loss_type == 9:
        coco_mode = True

    label_name = 'softmax_label'
    label_shape = (args.batch_size,)
    if args.output_c2c:
        label_shape = (args.batch_size, 2)
    if data_extra is None:
        model = mx.mod.Module(
            context=ctx,
            symbol=sym,
        )
    else:
        data_names = ('data', 'extra')
        # label_name = ''
        model = mx.mod.Module(
            context=ctx,
            symbol=sym,
            data_names=data_names,
            label_names=(label_name,),
        )

    if args.use_val:
        val_dataiter = FaceImageIter(
            batch_size=args.batch_size,
            data_shape=data_shape,
            path_imgrec=val_rec,
            shuffle=False,
            rand_mirror=False,
            mean=mean,
            ctx_num=args.ctx_num,
            data_extra=data_extra,
        )
    else:
        val_dataiter = None

    if len(data_dir_list) == 1 and args.loss_type != 12 and args.loss_type != 13:
        train_dataiter = FaceImageIter(
            batch_size=args.batch_size,
            data_shape=data_shape,
            path_imgrec=path_imgrec,
            shuffle=True,
            rand_mirror=args.rand_mirror,
            mean=mean,
            cutoff=args.cutoff,
            c2c_threshold=args.c2c_threshold,
            output_c2c=args.output_c2c,
            c2c_mode=args.c2c_mode,
            ctx_num=args.ctx_num,
            images_per_identity=args.images_per_identity,
            data_extra=data_extra,
            hard_mining=hard_mining,
            triplet_params=triplet_params,
            coco_mode=coco_mode,
            mx_model=model,
            label_name=label_name,
        )
    else:
        iter_list = []
        for _data_dir in data_dir_list:
            _path_imgrec = os.path.join(_data_dir, "train.rec")
            _dataiter = FaceImageIter(
                batch_size=args.batch_size,
                data_shape=data_shape,
                path_imgrec=_path_imgrec,
                shuffle=True,
                rand_mirror=args.rand_mirror,
                mean=mean,
                cutoff=args.cutoff,
                c2c_threshold=args.c2c_threshold,
                output_c2c=args.output_c2c,
                c2c_mode=args.c2c_mode,
                ctx_num=args.ctx_num,
                images_per_identity=args.images_per_identity,
                data_extra=data_extra,
                hard_mining=hard_mining,
                triplet_params=triplet_params,
                coco_mode=coco_mode,
                mx_model=model,
                label_name=label_name,
            )
            iter_list.append(_dataiter)
        iter_list.append(_dataiter)
        train_dataiter = FaceImageIterList(iter_list)

    if args.loss_type < 10:
        _metric = AccMetric()
    else:
        _metric = LossValueMetric()
    eval_metrics = [mx.metric.create(_metric)]

    if args.network[0] == 'r':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)  # resnet style
    elif args.network[0] == 'i' or args.network[0] == 'x':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)  # inception
    else:
        initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0 / args.ctx_num
    if args.noise_sgd > 0.0:
        print('use noise sgd')
        opt = NoiseSGD(scale=args.noise_sgd, learning_rate=base_lr, momentum=base_mom, wd=base_wd,
                       rescale_grad=_rescale)
    else:
        opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 20
    if args.loss_type == 12 or args.loss_type == 13:
        som = 2
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join(data_dir, name + ".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            print('ver', name)

    def ver_test(nbatch):
        results = []
        for i in range(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size,
                                                                               data_extra, label_shape)
            print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
            # print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
            results.append(acc2)
        return results

    def val_test():
        acc = AccMetric()
        val_metric = mx.metric.create(acc)
        val_metric.reset()
        val_dataiter.reset()
        for i, eval_batch in enumerate(val_dataiter):
            model.forward(eval_batch, is_train=False)
            model.update_metric(val_metric, eval_batch.label)
        acc_value = val_metric.get_name_value()[0][1]
        print('VACC: %f' % (acc_value))

    highest_acc = [0.0, 0.0]  # lfw and target
    # for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps) == 0:
        lr_steps = [40000, 60000, 80000]
        if args.loss_type >= 1 and args.loss_type <= 5:
            lr_steps = [100000, 140000, 160000]
        p = 512.0 / args.batch_size
        for l in range(len(lr_steps)):
            lr_steps[l] = int(lr_steps[l] * p)
    else:
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)

    def _batch_callback(param):
        # global global_step
        global_step[0] += 1
        mbatch = global_step[0]
        for _lr in lr_steps:
            if mbatch == args.beta_freeze + _lr:
                opt.lr *= 0.1
                print('lr change to', opt.lr)
                break

        _cb(param)
        if mbatch % 1000 == 0:
            print('lr-batch-epoch:', opt.lr, param.nbatch, param.epoch)

        if mbatch >= 0 and mbatch % args.verbose == 0:
            acc_list = ver_test(mbatch)
            save_step[0] += 1
            msave = save_step[0]
            do_save = False
            if len(acc_list) > 0:
                lfw_score = acc_list[0]
                if lfw_score > highest_acc[0]:
                    highest_acc[0] = lfw_score
                    if lfw_score >= 0.998:
                        do_save = True
                if acc_list[-1] >= highest_acc[-1]:
                    highest_acc[-1] = acc_list[-1]
                    if lfw_score >= 0.99:
                        do_save = True
            if args.ckpt == 0:
                do_save = False
            elif args.ckpt > 1:
                do_save = True
            # for i in xrange(len(acc_list)):
            #  acc = acc_list[i]
            #  if acc>=highest_acc[i]:
            #    highest_acc[i] = acc
            #    if lfw_score>=0.99:
            #      do_save = True
            # if args.loss_type==1 and mbatch>lr_steps[-1] and mbatch%10000==0:
            #  do_save = True
            if do_save:
                print('saving', msave)
                if val_dataiter is not None:
                    val_test()
                arg, aux = model.get_params()
                mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
                # if acc>=highest_acc[0]:
                #  lfw_npy = "%s-lfw-%04d" % (prefix, msave)
                #  X = np.concatenate(embeddings_list, axis=0)
                #  print('saving lfw npy', X.shape)
                #  np.save(lfw_npy, X)
            print('[%d]Accuracy-Highest: %1.5f' % (mbatch, highest_acc[-1]))
        if mbatch <= args.beta_freeze:
            _beta = args.beta
        else:
            move = max(0, mbatch - args.beta_freeze)
            _beta = max(args.beta_min, args.beta * math.pow(1 + args.gamma * move, -1.0 * args.power))
        # print('beta', _beta)
        os.environ['BETA'] = str(_beta)
        if args.max_steps > 0 and mbatch > args.max_steps:
            sys.exit(0)

    def _epoch_callback(epoch, sym, arg_params, aux_params):
        nonlocal highest_accT, best_epoch

        train_dataiter.reset()
        acc_metric = AccMetric()
        eval_metric = mx.metric.create(acc_metric)

        for batch in train_dataiter:
            model.forward(batch, is_train=False)
            model.update_metric(eval_metric, batch.label)

        raw_value = eval_metric.get_name_value()[0][1]
        current_acc = float(raw_value[0]) if isinstance(raw_value, list) else float(raw_value)

        if current_acc > highest_accT:
            highest_accT = current_acc
            best_epoch = epoch

            # Guardar usando nombre fijo para los pesos: "model.params"
            model_prefix = args.prefix  # Ej. "./checkpoints/model"
            param_filename = f"{model_prefix}.params"
            symbol_filename = f"{model_prefix}-symbol.json"

            # Guardar los parámetros (params) y el símbolo (opcional)
            save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
            save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
            mx.nd.save(param_filename, save_dict)

            sym.save(symbol_filename)

            print(f'✅ [Época {epoch}] Accuracy mejorado: {current_acc:.4f} (Guardado como model.params)')
        else:
            print(f'✖ [Época {epoch}] Accuracy actual: {current_acc:.4f} (Mejor: {highest_accT:.4f})')

    # epoch_cb = mx.callback.do_checkpoint(prefix, 1)

    # def _epoch_callback(epoch, sym, arg_params, aux_params):
    #  print('epoch-end', epoch)



    model.fit(train_dataiter,
              begin_epoch=begin_epoch,
              num_epoch=end_epoch,
              eval_data=val_dataiter,
              eval_metric=eval_metrics,
              kvstore='device',
              optimizer=opt,
              # optimizer_params   = optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              batch_end_callback=_batch_callback,
              epoch_end_callback=_epoch_callback)

    arg_params, aux_params = model.get_params()
    mx.model.save_checkpoint(args.prefix, args.end_epoch, model.symbol, arg_params, aux_params)

def main():
    # time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
