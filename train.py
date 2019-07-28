import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()

    # for writing loss to tensorboard
    if opt.tensorboardx:
        run_folder = os.path.join(os.path.join(opt.checkpoints_dir, opt.name),str(opt.id))
        if not os.path.isdir(run_folder):
            os.makedirs(run_folder)
        writer = SummaryWriter(log_dir=run_folder)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): # epoch: 1 ~ (100+100+1)
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset): # 1759 images
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0: # print loss info each 100 iterations
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size # batch_size is 1
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

                # write loss curve to tensorboard
                if opt.tensorboardx:
                    # combined verison (this should be correct)
                    writer.add_scalars('loss/train_loss', losses, total_steps)

                    # seperate version
                    for k, v in losses.items():
                        writer.add_scalar('loss/{}'.format(k), v, total_steps)

                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
