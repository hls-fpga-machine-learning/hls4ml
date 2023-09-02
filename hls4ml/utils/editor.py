import tempfile

import PySimpleGUI as sg

from .plot import plot_model

SG_THEME = 'SystemDefault'

sg.theme(SG_THEME)


def edit_model_configuration(model):
    arch_file = tempfile.NamedTemporaryFile(suffix='.png')
    plot_model(model, to_file=arch_file.name, show_shapes=True, show_precision=True)

    config_rows = []

    current_config = {}

    for layer in model.graph.values():
        if layer.class_name == 'Input':
            # We skip the Input layer since changing the result_t of input layers isn't allowed.
            continue
        config_attrs = [a for a in layer.expected_attributes if a.configurable]
        if len(config_attrs) > 0:
            layer_rows = []
            for attr in config_attrs:
                attr_val = layer.get_attr(attr.name, default='')
                if attr.name.endswith('_t'):
                    attr_val = attr_val.precision

                # Use a combo for bool and Choice attributes
                input_key = layer.name + '!#!' + attr.name
                if attr.value_type == bool:
                    attr_val = str(attr_val)
                    input_column = sg.Combo(
                        values=['True', 'False'], default_value=attr_val, key=input_key, size=23, enable_events=True
                    )
                elif attr.__class__.__name__ == 'ChoiceAttribute':  # Avoids importing attributes
                    input_column = sg.Combo(
                        values=attr.choices, default_value=attr_val, key=input_key, size=23, enable_events=True
                    )
                else:
                    attr_val = str(attr_val)
                    input_column = sg.Input(default_text=attr_val, key=input_key, size=25, enable_events=True)

                # Save current config
                current_config[input_key] = attr_val

                attr_columns = [
                    sg.Text(attr.name, size=25),
                    input_column,
                ]

                layer_rows.append(attr_columns)

            layer_frame = sg.Frame(layer.name + ' (' + layer.class_name + ')', layer_rows)
            layer_column = sg.Column([[layer_frame]])
            config_rows.append([layer_column])

    image_column = sg.Column([[sg.Image(filename=arch_file.name, key='!#!_image')]], scrollable=True)
    config_column = sg.Column(config_rows, scrollable=True, vertical_scroll_only=True)

    content_row = [image_column, config_column]

    buttons_row = [sg.Text('', key='!#!_info'), sg.Push(), sg.Button('Update'), sg.Button('Close')]

    layout = [
        content_row,
        buttons_row,
    ]

    # Create the window
    window = sg.Window('HLS4ML Configuration Editor', layout, resizable=True, finalize=True)

    image_column.expand(True, True)

    # Create an event loop
    while True:
        event, new_config = window.read()
        if event == 'Close' or event == sg.WIN_CLOSED:
            break
        if event == 'Update':
            _update_model_config(model, current_config, new_config)
            plot_model(model, to_file=arch_file.name, show_shapes=True, show_precision=True)
            window['!#!_image'].update(filename=arch_file.name, visible=True)
            window['!#!_info'].update('Configuration updated.')
            window.refresh()
        if '!#!' in event:
            window['!#!_info'].update('')
            window.refresh()

    try:
        arch_file.close()
    except Exception:
        pass
    window.close()


def _update_model_config(model, current_config, new_config):
    from hls4ml.model.types import NamedType

    changes_made = False
    for key, new_val_str in new_config.items():
        # Only update if changes were made
        if current_config[key] == new_val_str:
            continue

        changes_made = True
        layer_name, attr_name = key.split('!#!')
        layer = model.graph[layer_name]

        if attr_name.endswith('_t'):
            # This is a bit of a hack until we have a more robust configuration handling.
            # Essentially we will replace the NamedType attribute of the layer, but we also have to update the corresponding
            # variables that used the old types. While doing so, we have to ensure updated precision is bound to a new type,
            # so as to avoid overriding model_default_t, except for result_t, which will have a name layerX_t (X being the
            # index of the layer).
            new_precision = model.config.backend.convert_precision_string(new_val_str)
            old_named_type = layer.get_attr(attr_name)
            if attr_name == 'result_t':
                type_name = old_named_type.name
            else:
                type_name = layer.name + '_' + attr_name
            new_named_type = NamedType(type_name, new_precision)

            # Update the variables with the new type
            for var in layer.variables.values():
                if var.type is old_named_type:
                    var.type = new_named_type

            # Update the weights with the new type
            for w in layer.weights.values():
                if w.type is old_named_type:
                    w.type = new_named_type

            layer.set_attr(attr_name, new_named_type)  # Ensure the type is updated
        else:
            old_val = layer.get_attr(attr_name)
            attr_type = type(old_val)
            new_val = _parse_type(attr_type, new_val_str)
            if new_val is not None:
                layer.set_attr(attr_name, new_val)

    if changes_made:
        # Reapply the types flow (to convert from e.g., FixedPrecisionType to APFixedPrecisionType)
        backend_name = model.config.backend.name.lower()
        # For now, all backends have these flows, in the future we will have to trigger this differently
        # TODO Don't rely on names of flows to update configuration
        model.apply_flow(f'{backend_name}:specific_types')
        model.apply_flow(f'{backend_name}:apply_templates')


def _parse_type(attr_type, new_val_str):
    if attr_type == int:
        return attr_type(new_val_str)
    elif attr_type == str:
        return new_val_str
    elif attr_type == bool:
        bool_map = {
            'true': True,
            '1': True,
            'false': False,
            '0': False,
            True: True,
            False: False,
        }
        if new_val_str.lower() in bool_map:
            return bool_map[new_val_str.lower()]

    # Otherwise
    print('WARNING: Cannot convert string "{new_val_str}" to type {attr_type}')
    return None
