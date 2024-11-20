import tikzplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import torch
import pickle
import json
from lib.test.evaluation.environment import env_settings
from lib.test.analysis.extract_results import extract_results


def get_plot_draw_styles():
    plot_draw_style = [
                    #    {'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                    #    {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
                       {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                       {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                       {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                       {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                       {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                       {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                       {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                       {'color': (0.7, 0.6, 0.2), 'line_style': '-'}]

    return plot_draw_style


def check_eval_data_is_valid(eval_data, trackers, dataset):
    """ Checks if the pre-computed results are valid"""
    seq_names = [s.name for s in dataset]
    seq_names_saved = eval_data['sequences']

    tracker_names_f = [(t.name, t.parameter_name, t.run_id) for t in trackers]
    tracker_names_f_saved = [(t['name'], t['param'], t['run_id']) for t in eval_data['trackers']]

    return seq_names == seq_names_saved and tracker_names_f == tracker_names_f_saved


def merge_multiple_runs(eval_data):
    new_tracker_names = []
    ave_success_rate_plot_overlap_merged = []
    ave_success_rate_plot_center_merged = []
    ave_success_rate_plot_center_norm_merged = []
    avg_overlap_all_merged = []

    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all'])

    trackers = eval_data['trackers']
    merged = torch.zeros(len(trackers), dtype=torch.uint8)
    for i in range(len(trackers)):
        if merged[i]:
            continue
        base_tracker = trackers[i]
        new_tracker_names.append(base_tracker)

        match = [t['name'] == base_tracker['name'] and t['param'] == base_tracker['param'] for t in trackers]
        match = torch.tensor(match)

        ave_success_rate_plot_overlap_merged.append(ave_success_rate_plot_overlap[:, match, :].mean(1))
        ave_success_rate_plot_center_merged.append(ave_success_rate_plot_center[:, match, :].mean(1))
        ave_success_rate_plot_center_norm_merged.append(ave_success_rate_plot_center_norm[:, match, :].mean(1))
        avg_overlap_all_merged.append(avg_overlap_all[:, match].mean(1))

        merged[match] = 1

    ave_success_rate_plot_overlap_merged = torch.stack(ave_success_rate_plot_overlap_merged, dim=1)
    ave_success_rate_plot_center_merged = torch.stack(ave_success_rate_plot_center_merged, dim=1)
    ave_success_rate_plot_center_norm_merged = torch.stack(ave_success_rate_plot_center_norm_merged, dim=1)
    avg_overlap_all_merged = torch.stack(avg_overlap_all_merged, dim=1)

    eval_data['trackers'] = new_tracker_names
    eval_data['ave_success_rate_plot_overlap'] = ave_success_rate_plot_overlap_merged.tolist()
    eval_data['ave_success_rate_plot_center'] = ave_success_rate_plot_center_merged.tolist()
    eval_data['ave_success_rate_plot_center_norm'] = ave_success_rate_plot_center_norm_merged.tolist()
    eval_data['avg_overlap_all'] = avg_overlap_all_merged.tolist()

    return eval_data


def get_tracker_display_name(tracker):
    if tracker['disp_name'] is None:
        if tracker['run_id'] is None:
            disp_name = '{}_{}'.format(tracker['name'], tracker['param'])
        else:
            disp_name = '{}_{}_{:03d}'.format(tracker['name'], tracker['param'],
                                              tracker['run_id'])
    else:
        disp_name = tracker['disp_name']

    return  disp_name


def plot_draw_save(y, x, scores, trackers, plot_draw_styles, result_plot_path, plot_opts):
    plt.rcParams['text.usetex']=True
    plt.rcParams["font.family"] = "Times New Roman"
    # Plot settings
    font_size = plot_opts.get('font_size', 25)
    font_size_axis = plot_opts.get('font_size_axis', 20)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 15)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']
    if 'attr' in plot_opts:
        attr = plot_opts['attr']
    else:
        attr = None

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    ylabel = "%s"%(ylabel.replace('%','\%'))
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = r"\textbf{%s}" %(plot_opts['title'])
    print

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = scores.argsort(descending=False)

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        if trackers[id_sort]['disp_name'].startswith('SAMURAI'):
            alpha = 1.0
            line_style = '-'
            if trackers[id_sort]['disp_name'] == 'SAMURAI-L':
                color = (1.0, 0.0, 0.0)
            elif trackers[id_sort]['disp_name'] == 'SAMURAI-B':
                color = (0.0, 0.0, 1.0)
        elif trackers[id_sort]['disp_name'].startswith('SAM2.1'):
            alpha = 0.8
            line_style = '--'
            if trackers[id_sort]['disp_name'] == 'SAM2.1-L':
                color = (1.0, 0.0, 0.0)
            elif trackers[id_sort]['disp_name'] == 'SAM2.1-B':
                color = (0.0, 0.0, 1.0)
        else:
            alpha = 0.5
            color = plot_draw_styles[index_sort.numel() - id - 1]['color']
            line_style = ":"
        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=line_width,
                       color=color,
                       linestyle=line_style,
                       alpha=alpha)

        plotted_lines.append(line[0])

        tracker = trackers[id_sort]
        disp_name = get_tracker_display_name(tracker)

        legend_text.append('{} [{:.1f}]'.format(disp_name, scores[id_sort]))

    try:
        # add bold to top method
        # for i in range(1,2):
        #     legend_text[-i] = r'\textbf{%s}'%(legend_text[-i])

        for id, id_sort in enumerate(index_sort):
            if trackers[id_sort]['disp_name'].startswith('SAMTrack'):
                legend_text[id] = r'\textbf{%s}'%(legend_text[id])

        ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
                  fontsize=font_size_legend, framealpha=1.0)
    except:
        pass

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           xlim=xlim, ylim=ylim,
           title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    def tikzplotlib_fix_ncols(obj):
        """
        workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
        """
        if hasattr(obj, "_ncols"):
            obj._ncol = obj._ncols
        for child in obj.get_children():
            tikzplotlib_fix_ncols(child)
        
    tikzplotlib_fix_ncols(fig)

    # tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type))
    if attr is not None:
        fig.savefig('{}/{}_{}_plot.pdf'.format(result_plot_path, plot_type, attr), dpi=300, format='pdf', transparent=True)
    else:
        fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    plt.draw()


def check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation=False, **kwargs):
    # Load data
    settings = env_settings()

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data_path = os.path.join(result_plot_path, 'eval_data.pkl')

    if os.path.isfile(eval_data_path) and not force_evaluation:
        with open(eval_data_path, 'rb') as fh:
            eval_data = pickle.load(fh)
    else:
        # print('Pre-computed evaluation data not found. Computing results!')
        eval_data = extract_results(trackers, dataset, report_name, **kwargs)

    if not check_eval_data_is_valid(eval_data, trackers, dataset):
        # print('Pre-computed evaluation data invalid. Re-computing results!')
        eval_data = extract_results(trackers, dataset, report_name, **kwargs)
        # pass
    else:
        # Update display names
        tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name}
                         for t in trackers]
        eval_data['trackers'] = tracker_names
    with open(eval_data_path, 'wb') as fh:
        pickle.dump(eval_data, fh)
    return eval_data


def get_auc_curve(ave_success_rate_plot_overlap, valid_sequence):
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)

    return auc_curve, auc


def get_prec_curve(ave_success_rate_plot_center, valid_sequence):
    ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, 20]

    return prec_curve, prec_score

def plot_per_attribute_results(trackers, dataset, report_name, merge_results=False,
                                plot_types=('success'), **kwargs):
    # Load data
    settings = env_settings()

    plot_draw_styles = get_plot_draw_styles()

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    attr_folder = 'data/LaSOT/att'

    attr_list = ['Illumination Variation', 'Partial Occlusion', 'Deformation', 'Motion Blur', 'Camera Motion', 'Rotation', 'Background Clutter', 'Viewpoint Change', 'Scale Variation', 'Full Occlusion', 'Fast Motion', 'Out-of-View', 'Low Resolution', 'Aspect Ration Change']
    attr_list = ['IV', 'POC', 'DEF', 'MB', 'CM', 'ROT', 'BC', 'VC', 'SV', 'FOC', 'FM', 'OV', 'LR', 'ARC']

    # Iterate over the sequence and construct a valid_sequence for each attribute
    valid_sequence_attr = {}
    for attr in attr_list:
        valid_sequence_attr[attr] = torch.zeros(valid_sequence.shape[0], dtype=torch.bool)
    for seq_id, seq_obj in enumerate(dataset):
        seq_name = seq_obj.name
        attr_txt = osp.join(attr_folder, f'{seq_name}.txt')
        if osp.exists(attr_txt):
            # read the attribute file into a list of True and False
            # the attribute file looks like this: 0,0,0,0,0,1,0,1,1,0,0,0,0,0
            attr_anno = np.loadtxt(attr_txt, dtype=int, delimiter=',')
            # broadcast the valid_sequence to the attribute list
            for attr_id, attr in enumerate(attr_list):
                valid_sequence_attr[attr][seq_id] = attr_anno[attr_id]
        else:
            raise Exception(f'Attribute file not found for sequence {seq_name}')

    tracker_names = eval_data['trackers']

    if report_name == 'LaSOT-ext':
        ylim_success = (0, 95)
        ylim_precision = (0, 85)
        ylim_norm_precision = (0, 85)
        report_name = "LaSOT_{ext}"
    else:
        ylim_success = (0, 95)
        ylim_precision = (0, 100)
        ylim_norm_precision = (0, 88)
    
    threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    for attr in attr_list:
        scores = {}

        print(f'{attr}: {valid_sequence_attr[attr].sum().item()}')
        valid_sequence_attr[attr] = valid_sequence_attr[attr] & valid_sequence
        
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence_attr[attr])
        scores['AUC'] = auc

        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence_attr[attr])
        scores['Precision'] = prec_score

        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence_attr[attr])
        scores['Norm Precision'] = norm_prec_score

        tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
        report_text = generate_formatted_report(tracker_disp_names, scores, table_name=attr)
        print(report_text)

        success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold', 'attr': attr,
                             'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': ylim_success, 'title': f'Success\ of\ {attr}\ ({report_name})'}
        plot_draw_save(auc_curve, threshold_set_overlap, auc, tracker_names, plot_draw_styles, result_plot_path, success_plot_opts)

    


def plot_results(trackers, dataset, report_name, merge_results=False,
                 plot_types=('success'), force_evaluation=False, **kwargs):
    """
    Plot results for the given trackers

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success',
                    'prec' (precision), and 'norm_prec' (normalized precision)
    """
    # Load data
    settings = env_settings()

    plot_draw_styles = get_plot_draw_styles()

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']

    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nPlotting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    print('\nGenerating plots for: {}'.format(report_name))

    print(report_name)
    if report_name == 'LaSOT':
        ylim_success = (0, 95)
        ylim_precision = (0, 95)
        ylim_norm_precision = (0, 95)
    elif report_name == 'LaSOT-ext':
        ylim_success = (0, 85)
        ylim_precision = (0, 85)
        ylim_norm_precision = (0, 85)
    else:
        ylim_success = (0, 85)
        ylim_precision = (0, 85)
        ylim_norm_precision = (0, 85)

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])

        success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                             'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': ylim_success, 'title': f'Success\ ({report_name})'}
        plot_draw_save(auc_curve, threshold_set_overlap, auc, tracker_names, plot_draw_styles, result_plot_path, success_plot_opts)

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        threshold_set_center = torch.tensor(eval_data['threshold_set_center'])

        precision_plot_opts = {'plot_type': 'precision', 'legend_loc': 'lower right',
                               'xlabel': 'Location error threshold [pixels]', 'ylabel': 'Distance Precision [%]',
                               'xlim': (0, 50), 'ylim': ylim_precision, 'title': f'Precision\ ({report_name})'}
        plot_draw_save(prec_curve, threshold_set_center, prec_score, tracker_names, plot_draw_styles, result_plot_path,
                       precision_plot_opts)

    # ********************************  Norm Precision Plot **************************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        threshold_set_center_norm = torch.tensor(eval_data['threshold_set_center_norm'])

        norm_precision_plot_opts = {'plot_type': 'norm_precision', 'legend_loc': 'lower right',
                                    'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision [%]',
                                    'xlim': (0, 0.5), 'ylim': ylim_norm_precision, 'title': f'Normalized\ Precision\ ({report_name})'}
        plot_draw_save(prec_curve, threshold_set_center_norm, prec_score, tracker_names, plot_draw_styles, result_plot_path,
                       norm_precision_plot_opts)

    plt.show()


def generate_formatted_report(row_labels, scores, table_name=''):
    name_width = max([len(d) for d in row_labels] + [len(table_name)]) + 5
    min_score_width = 10

    report_text = '\n{label: <{width}} |'.format(label=table_name, width=name_width)

    score_widths = [max(min_score_width, len(k) + 3) for k in scores.keys()]

    for s, s_w in zip(scores.keys(), score_widths):
        report_text = '{prev} {s: <{width}} |'.format(prev=report_text, s=s, width=s_w)

    report_text = '{prev}\n'.format(prev=report_text)

    for trk_id, d_name in enumerate(row_labels):
        # display name
        report_text = '{prev}{tracker: <{width}} |'.format(prev=report_text, tracker=d_name,
                                                           width=name_width)
        for (score_type, score_value), s_w in zip(scores.items(), score_widths):
            report_text = '{prev} {score: <{width}} |'.format(prev=report_text,
                                                              score='{:0.2f}'.format(score_value[trk_id].item()),
                                                              width=s_w)
        report_text = '{prev}\n'.format(prev=report_text)

    return report_text

def print_per_attribute_results(trackers, dataset, report_name, merge_results=False,
                                plot_types=('success'), **kwargs):
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    attr_folder = 'data/LaSOT/att'

    attr_list = ['Illumination Variation', 'Partial Occlusion', 'Deformation', 'Motion Blur', 'Camera Motion', 'Rotation', 'Background Clutter', 'Viewpoint Change', 'Scale Variation', 'Full Occlusion', 'Fast Motion', 'Out-of-View', 'Low Resolution', 'Aspect Ration Change']
    attr_list = ['IV', 'POC', 'DEF', 'MB', 'CM', 'ROT', 'BC', 'VC', 'SV', 'FOC', 'FM', 'OV', 'LR', 'ARC']

    # Iterate over the sequence and construct a valid_sequence for each attribute
    valid_sequence_attr = {}
    for attr in attr_list:
        valid_sequence_attr[attr] = torch.zeros(valid_sequence.shape[0], dtype=torch.bool)
    for seq_id, seq_obj in enumerate(dataset):
        seq_name = seq_obj.name
        attr_txt = osp.join(attr_folder, f'{seq_name}.txt')
        if osp.exists(attr_txt):
            # read the attribute file into a list of True and False
            # the attribute file looks like this: 0,0,0,0,0,1,0,1,1,0,0,0,0,0
            attr_anno = np.loadtxt(attr_txt, dtype=int, delimiter=',')
            # broadcast the valid_sequence to the attribute list
            for attr_id, attr in enumerate(attr_list):
                valid_sequence_attr[attr][seq_id] = attr_anno[attr_id]
        else:
            raise Exception(f'Attribute file not found for sequence {seq_name}')

    tracker_names = eval_data['trackers']
    
    threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    for attr in attr_list:
        scores = {}

        print(f'{attr}: {valid_sequence_attr[attr].sum().item()}')
        valid_sequence_attr[attr] = valid_sequence_attr[attr] & valid_sequence
        
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence_attr[attr])
        scores['AUC'] = auc

        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence_attr[attr])
        scores['Precision'] = prec_score

        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence_attr[attr])
        scores['Norm Precision'] = norm_prec_score

        tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
        report_text = generate_formatted_report(tracker_disp_names, scores, table_name=attr)
        print(report_text)


def print_results(trackers, dataset, report_name, merge_results=False,
                  plot_types=('success'), **kwargs):
    """ Print the results for the given trackers in a formatted table
    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
                    'prec' (prints precision score), and 'norm_prec' (prints normalized precision score)
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    scores = {}

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    # Print
    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
    print(report_text)


def plot_got_success(trackers, report_name):
    """ Plot success plot for GOT-10k dataset using the json reports.
    Save the json reports from http://got-10k.aitestunion.com/leaderboard in the directory set to
    env_settings.got_reports_path

    The tracker name in the experiment file should be set to the name of the report file for that tracker,
    e.g. DiMP50_report_2019_09_02_15_44_25 if the report is name DiMP50_report_2019_09_02_15_44_25.json

    args:
        trackers - List of trackers to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
    """
    # Load data
    settings = env_settings()
    plot_draw_styles = get_plot_draw_styles()

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    auc_curve = torch.zeros((len(trackers), 101))
    scores = torch.zeros(len(trackers))

    # Load results
    tracker_names = []
    for trk_id, trk in enumerate(trackers):
        json_path = '{}/{}.json'.format(settings.got_reports_path, trk.name)

        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                eval_data = json.load(f)
        else:
            raise Exception('Report not found {}'.format(json_path))

        if len(eval_data.keys()) > 1:
            raise Exception

        # First field is the tracker name. Index it out
        eval_data = eval_data[list(eval_data.keys())[0]]
        if 'succ_curve' in eval_data.keys():
            curve = eval_data['succ_curve']
            ao = eval_data['ao']
        elif 'overall' in eval_data.keys() and 'succ_curve' in eval_data['overall'].keys():
            curve = eval_data['overall']['succ_curve']
            ao = eval_data['overall']['ao']
        else:
            raise Exception('Invalid JSON file {}'.format(json_path))

        auc_curve[trk_id, :] = torch.tensor(curve) * 100.0
        scores[trk_id] = ao * 100.0

        tracker_names.append({'name': trk.name, 'param': trk.parameter_name, 'run_id': trk.run_id,
                              'disp_name': trk.display_name})

    threshold_set_overlap = torch.arange(0.0, 1.01, 0.01, dtype=torch.float64)

    success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                         'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 100), 'title': 'Success plot'}
    plot_draw_save(auc_curve, threshold_set_overlap, scores, tracker_names, plot_draw_styles, result_plot_path,
                   success_plot_opts)
    plt.show()


def print_per_sequence_results(trackers, dataset, report_name, merge_results=False,
                               filter_criteria=None, **kwargs):
    """ Print per-sequence results for the given trackers. Additionally, the sequences to list can be filtered using
    the filter criteria.

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        filter_criteria - Filter sequence results which are reported. Following modes are supported
                        None: No filtering. Display results for all sequences in dataset
                        'ao_min': Only display sequences for which the minimum average overlap (AO) score over the
                                  trackers is less than a threshold filter_criteria['threshold']. This mode can
                                  be used to select sequences where at least one tracker performs poorly.
                        'ao_max': Only display sequences for which the maximum average overlap (AO) score over the
                                  trackers is less than a threshold filter_criteria['threshold']. This mode can
                                  be used to select sequences all tracker performs poorly.
                        'delta_ao': Only display sequences for which the performance of different trackers vary by at
                                    least filter_criteria['threshold'] in average overlap (AO) score. This mode can
                                    be used to select sequences where the behaviour of the trackers greatly differ
                                    between each other.
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    sequence_names = eval_data['sequences']
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all']) * 100.0

    # Filter sequences
    if filter_criteria is not None:
        if filter_criteria['mode'] == 'ao_min':
            min_ao = avg_overlap_all.min(dim=1)[0]
            valid_sequence = valid_sequence & (min_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'ao_max':
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & (max_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'delta_ao':
            min_ao = avg_overlap_all.min(dim=1)[0]
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & ((max_ao - min_ao) > filter_criteria['threshold'])
        else:
            raise Exception

    avg_overlap_all = avg_overlap_all[valid_sequence, :]
    sequence_names = [s + ' (ID={})'.format(i) for i, (s, v) in enumerate(zip(sequence_names, valid_sequence.tolist())) if v]

    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]

    scores_per_tracker = {k: avg_overlap_all[:, i] for i, k in enumerate(tracker_disp_names)}
    report_text = generate_formatted_report(sequence_names, scores_per_tracker)

    print(report_text)


def print_results_per_video(trackers, dataset, report_name, merge_results=False,
                  plot_types=('success'), per_video=False, **kwargs):
    """ Print the results for the given trackers in a formatted table
    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
                    'prec' (prints precision score), and 'norm_prec' (prints normalized precision score)
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    seq_lens = len(eval_data['sequences'])
    eval_datas = [{} for _ in range(seq_lens)]
    if per_video:
        for key, value in eval_data.items():
            if len(value) == seq_lens:
                for i in range(seq_lens):
                    eval_datas[i][key] = [value[i]]
            else:
                for i in range(seq_lens):
                    eval_datas[i][key] = value

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    scores = {}

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    # Print
    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
    print(report_text)

    if per_video:
        for i in range(seq_lens):
            eval_data = eval_datas[i]

            print('\n{} sequences'.format(eval_data['sequences'][0]))

            scores = {}
            valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

            # ********************************  Success Plot **************************************
            if 'success' in plot_types:
                threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
                ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

                # Index out valid sequences
                auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
                scores['AUC'] = auc
                scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
                scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

            # ********************************  Precision Plot **************************************
            if 'prec' in plot_types:
                ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

                # Index out valid sequences
                prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
                scores['Precision'] = prec_score

            # ********************************  Norm Precision Plot *********************************
            if 'norm_prec' in plot_types:
                ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

                # Index out valid sequences
                norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
                scores['Norm Precision'] = norm_prec_score

            # Print
            tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
            report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
            print(report_text)
