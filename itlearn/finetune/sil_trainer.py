import torch

from itlearn.finetune.trainer import Trainer
from itlearn.finetune.agents import AgentsGumbel, AgentsA2C
from itlearn.finetune.imitate_utils import imitate_fr_en, imitate_en_de, get_fr_en_imitate_stats, \
    get_en_de_imitate_stats, finetune_en_de
import time
import os
from pandas import DataFrame
import matplotlib.pyplot as plt


def plot_imitate_stats(teacher_stats, imitate_statss):
    iterss = [res[0] for res in imitate_statss]
    statss = [res[1] for res in imitate_statss]
    fig, axs = plt.subplots(len(teacher_stats), figsize=(7, 7 * len(teacher_stats)))
    axs = axs.reshape(-1) if len(teacher_stats) > 1 else [axs]
    for name, ax in zip(teacher_stats, axs):
        student_vals = [stats[name] for stats in statss]
        teacher_val = teacher_stats[name]
        ax.plot(iterss, student_vals, label='student')
        ax.plot([iterss[0], iterss[-1]], [teacher_val, teacher_val], label='teacher')
        ax.set_title(name)
        ax.legend()
    return fig


class SILTrainer(Trainer):
    def __init__(self, args, model, train_it, dev_it, extra_input,
                 loss_cos, loss_names, monitor_names):
        super(SILTrainer, self).__init__(args, model, train_it, dev_it, extra_input,
                                         loss_cos, loss_names, monitor_names)
        use_sil = hasattr(args, 'k1') and hasattr(args, 'fr_en_k2') and hasattr(args, 'en_de_k2')
        assert use_sil

        # Prepare SIL
        # Prepare_init_student
        if 'gumbel' in args.setup:
            self.student = AgentsGumbel(args)
        else:
            self.student = AgentsA2C(args)
        if self.resume:
            self.student.load_state_dict(extra_input['resume']['student'])
        else:
            self.student.load_state_dict(model.state_dict())
        if torch.cuda.device_count() > 0 and args.gpu > -1:
            self.student.cuda(args.gpu)

        # Get opt fpr students
        self.stu_fr_en_opt, self.stu_en_de_opt = None, None
        if self.resume:
            self._get_student_opts(args)
            self.stu_fr_en_opt.load_state_dict(extra_input['resume']['stu_fr_en_opt'])
            self.stu_en_de_opt.load_state_dict(extra_input['resume']['stu_en_de_opt'])
        self.max_itlearn_steps = args.max_itlearn_steps if args.max_itlearn_steps > 0 else args.max_training_steps

    def _get_student_opts(self):
        # Same opt
        if self.args.same_opt and self.stu_opts[0] is not None and self.stu_opts[1] is not None:
            self.args.logger.info('Reuse optimizer!')

        # Create
        else:
            self.args.logger.info('Create new optimizer')
            self.stu_fr_en_opt = torch.optim.Adam(self.student.fr_en.parameters(), betas=(0.9, 0.98),
                                                  eps=1e-9, lr=self.args.fr_en_lr)
            self.stu_en_de_opt = torch.optim.Adam(self.student.en_de.parameters(), betas=(0.9, 0.98),
                                                  eps=1e-9, lr=self.args.en_de_lr)

    def train_step(self, iters, train_batch):
        self.selfplay_step(iters, train_batch)
        if (iters + 1) % self.args.k1 == 0 and (iters + 1) < self.max_itlearn_steps:
            self.args.logger.info('start imitating at iters {}'.format(iters + 1))
            self.sil_training(iters)

    def sil_training(self, iters):
        old_student_fr_en_stats = get_fr_en_imitate_stats(self.args, self.student, self.dev_it,
                                                          self.monitor_names, self.extra_input)
        old_student_en_de_stats = get_en_de_imitate_stats(self.args, self.student, self.dev_it)
        self._get_student_opts()

        start = time.time()
        self.model.eval()
        self.student.train()
        fr_en_statss = imitate_fr_en(self.args, student=self.student,
                                     teacher=self.model, train_it=self.train_it,
                                     dev_it=self.dev_it, monitor_names=self.monitor_names,
                                     extra_input=self.extra_input, opt=self.stu_fr_en_opt)
        end = time.time()
        self.args.logger.info('FrEn cost time: {:.2f}'.format(end - start))

        start = time.time()
        if not self.args.en_de_finetune:
            en_de_statss = imitate_en_de(self.args, student=self.student,
                                         teacher=self.model, train_it=self.train_it, dev_it=self.dev_it,
                                         opt=self.stu_en_de_opt, extra_input=self.extra_input)
        else:
            en_de_statss = finetune_en_de(self.args, student=self.student,
                                          teacher=self.model, train_it=self.train_it, dev_it=self.dev_it,
                                          opt=self.stu_en_de_opt, extra_input=self.extra_input)
        end = time.time()
        self.args.logger.info('EnDe cost time: {:.2f}'.format(end - start))

        # Report change of student and teacher
        self.model.eval()
        self.student.eval()
        teacher_fr_en_stats = get_fr_en_imitate_stats(self.args, self.model, self.dev_it,
                                                      self.monitor_names, self.extra_input)
        student_fr_en_stats = get_fr_en_imitate_stats(self.args, self.student, self.dev_it,
                                                      self.monitor_names, self.extra_input)
        teacher_en_de_stats = get_en_de_imitate_stats(self.args, self.model, self.dev_it)
        student_en_de_stats = get_en_de_imitate_stats(self.args, self.student, self.dev_it)
        df = DataFrame(columns=['teacher', 'stu', 'old_stu'])
        for name in student_fr_en_stats:
            df.loc[name, 'stu'] = student_fr_en_stats[name]
            df.loc[name, 'teacher'] = teacher_fr_en_stats[name]
            df.loc[name, 'old_stu'] = old_student_fr_en_stats[name]
        for name in student_en_de_stats:
            df.loc[name, 'stu'] = student_en_de_stats[name]
            df.loc[name, 'teacher'] = teacher_en_de_stats[name]
            df.loc[name, 'old_stu'] = old_student_en_de_stats[name]
        self.args.logger.info(str(df))

        if self.args.save_imitate_stats:
            print('Save imitation stats')
            if not os.path.exists(os.path.join(self.args.misc_path, self.args.id_str)):
                os.makedirs(os.path.join(self.args.misc_path, self.args.id_str))
            fr_en_fig = self.plot_imitate_stats(teacher_fr_en_stats, fr_en_statss)
            fr_en_fig.savefig(os.path.join(self.args.misc_path, self.args.id_str, 'fr_en_{}_stats.png'.format(iters + 1)))
            del fr_en_fig
            en_de_fig = self.plot_imitate_stats(teacher_en_de_stats, en_de_statss)
            en_de_fig.savefig(os.path.join(self.args.misc_path, self.args.id_str, 'en_de_{}_stats.png'.format(iters + 1)))
            del en_de_fig

        # Update teacher with finalized student
        self.model.load_state_dict(self.student.state_dict())
