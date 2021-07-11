class ConfigState:
    config = {
        'pixel_width': 0.25,
        'show_plots': True,
        'plotting_segments': 1,
        'create_plots': True,
        'save_figures': True,
        'print_stats': True,
        'write_csv': True,
        'autotrim_enabled': True,
        'autotrim_temperature': 80.0,
        'autotrim_percentage': 0.2,
        'manual_trim_longitudinal_start': 0,
        'manual_trim_longitudinal_end': 9999,
        'manual_trim_transversal_start': 0,
        'manual_trim_transversal_end': 9999,
        'roadwidth_threshold': 80.0,
        'roadwidth_adjust_left': 2,
        'roadwidth_adjust_right': 2,
        'lane_enabled': True,
        'lane_threshold': 110.0,
        'lane_to_use': 'warmest',
        'roller_detect_enabled': True,
        'roller_detect_temperature': 50,
        'roller_detect_interpolation': True,
        'gradient_enabled': True,
        'gradient_tolerance': 10.0,
        'moving_average_enabled': True,
        'moving_average_window': 100.0,
        'moving_average_percent': 90.0,
        'gradient_statistics_enabled': False,
        'cluster_npixels': 0,
        'cluster_sqm': 0.0,
        'tolerance': [5, 20, 1],
    }

    def update(self, job):
        mandatory_keys = ['title', 'file_path']
        for key in mandatory_keys:
            if key not in job:
                raise Exception(f'configuration field {key} must be set!')

        for key, value in job.items():
            self.config[key] = value

        return self.config

