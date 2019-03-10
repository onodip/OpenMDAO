"""
HTML file writing to create standalone XDSMjs output file.
"""
import base64
import json
import os

from six import itervalues

from openmdao.devtools.html_utils import read_files, write_div, head_and_body, write_script, \
    write_style

_DEFAULT_JSON_FILE = "xdsm.json"  # Used as default name if data is not embedded
_CHAR_SET = "utf-8"  # HTML character set

# Toolbar settings
_FONT_SIZES = [8, 9, 10, 11, 12, 13, 14]
_MODEL_HEIGHTS = [600, 650, 700, 750, 800, 850, 900, 950, 1000, 2000, 3000, 4000]

_HTML_TMP = (
    '<div style="margin-top:5px;">'
    '    <input type="text" id="awesompleteId" size="80" />'
    '    <button class="myButton" id="searchButtonId" title="Search"><i class="icon-search"></i></button>'
    '    <label id="searchCountId" style="display:inline-block;width:120px"></label>'
    '</div>'
    '<!--<select id="outputNamingSelect" onchange="OutputNameSelectChange()" style="display:none">-->'
    '<div id="currentPathId" class="path_string" style="display:none"></div>'
    '<div id="d3_content_div"></div>'
    '<div id="connectionId"></div>'
)


def write_html(outfile, source_data=None, data_file=None, embeddable=False, toolbar=True):
    """
    Writes XDSMjs HTML output file, with style and script files embedded.

    The source data can be the name of a JSON file or a dictionary.
    If a JSON file name is provided, the file will be referenced in the HTML.
    If the input is a dictionary, it will be embedded.

    If both data file and source data are given, data file is

    Parameters
    ----------
    outfile : str
        Output HTML file
    source_data : str or dict or None
        XDSM data in a dictionary or string
    data_file : str or None
        Output HTML file
    embeddable : bool, optional
        If True, gives a single HTML file that doesn't have the <html>, <DOCTYPE>, <body>
        and <head> tags. If False, gives a single, standalone HTML file for viewing.
    toolbar : bool
        Add diagram viewer toolbar
    """

    # directories
    main_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(main_dir, "XDSMjs")
    build_dir = os.path.join(code_dir, "build")
    style_dir = code_dir  # CSS

    with open(os.path.join(build_dir, "xdsm.bundle.js"), "r") as f:
        code = f.read()
        xdsm_bundle = write_script(code, {'type': 'text/javascript'})

    xdsm_attrs = {'class': 'xdsm'}
    # grab the data
    if data_file is not None:
        # Add name of the data file
        xdsm_attrs['data-mdo-file'] = data_file
    elif source_data is not None:
        if isinstance(source_data, (dict, str)):
            data_str = str(source_data)  # dictionary converted to string
        else:
            msg = ('Invalid data type for source data: {} \n'
                   'The source data should be a JSON file name or a dictionary.')
            raise ValueError(msg.format(type(source_data)))

        # Replace quote marks for the HTML syntax
        for i in ('u"', "u'", '"', "'"):  # quote marks and unicode prefixes
            data_str = data_str.replace(i, r'&quot;')
        xdsm_attrs['data-mdo'] = data_str
    else:  # both source data and data file name are None
        msg = 'Specify either "source_data" or "data_file".'
        raise ValueError(msg.format(type(source_data)))

    # grab the style

    # put all style and JS into index
    if toolbar:
        problem_viewer_dir = os.path.join(os.path.dirname(main_dir), "problem_viewer", "visualization")
        problem_viewer_style__dir = os.path.join(problem_viewer_dir, "style")
        problem_viewer_src_dir = os.path.join(problem_viewer_dir, "src")

        styles = read_files(('xdsm',), style_dir, 'css')
        styles.update(read_files(('partition_tree', 'awesomplete'), problem_viewer_style__dir, 'css'))
        with open(os.path.join(problem_viewer_style__dir, "fontello.woff"), "rb") as f:
            encoded_font = str(base64.b64encode(f.read()).decode("ascii"))
        src_names = 'constants', 'draw', 'legend', 'modal', 'ptN2', 'search', 'svg'
        srcs = read_files(src_names, problem_viewer_src_dir, 'js')
        scripts = '\n\n'.join([write_script(code, indent=4) for code in itervalues(srcs)])
        toolbar_div = write_div(content=["{{scripts}}", "{{title}}", "{{toolbar}}", "{{help}}",
                                         "{{magic}}"],
                                uid="ptN2ContentDivId")
    else:  # Default XDSMjs toolbar
        styles = read_files(('fontello', 'xdsm'), style_dir, 'css')
        toolbar_div = write_div(attrs={'class': 'xdsm-toolbar'})
    styles_elem = write_style(content='\n\n'.join(itervalues(styles)))
    xdsm_div = write_div(attrs=xdsm_attrs)
    body = '\n\n'.join([toolbar_div, xdsm_div])
    if toolbar:
        body = write_div(content=body, uid="all_pt_n2_content_div")

    if embeddable:
        index = '\n\n'.join([styles_elem, xdsm_bundle, body])
    else:
        meta = '<meta charset="{}">'.format(_CHAR_SET)

        head = '\n\n'.join([meta, styles_elem, xdsm_bundle])

        index = head_and_body(head, body, attrs={'class': "js", 'lang': ""})

    # Embed style, scripts and data to HTML
    with open(outfile, 'w') as f:
        f.write(index)

    # Replace references in the file
    if toolbar:
        from openmdao.devtools.html_utils import DiagramWriter

        h = DiagramWriter(filename=outfile,
                          title="OpenMDAO XDSM diagram.",
                          styles=styles, embeddable=True)

        # Toolbar
        toolbar = h.toolbar
        group1 = toolbar.add_button_group()
        group1.add_button("Return To Root", uid="returnToRootButtonId",
                          content="icon-home")
        group1.add_button("Back", uid="backButtonId", content="icon-left-big")
        group1.add_button("Forward", uid="forwardButtonId",
                          content="icon-right-big")
        group1.add_button("Up One Level", uid="upOneLevelButtonId",
                          content="icon-up-big")

        group2 = toolbar.add_button_group()
        group2.add_button("Uncollapse In View Only", uid="uncollapseInViewButtonId",
                          content="icon-resize-full")
        group2.add_button("Uncollapse All", uid="uncollapseAllButtonId",
                          content="icon-resize-full bigger-font")
        group2.add_button("Collapse Outputs In View Only", uid="collapseInViewButtonId",
                          content="icon-resize-small")
        group2.add_button("Collapse All Outputs", uid="collapseAllButtonId",
                          content="icon-resize-small bigger-font")
        group2.add_dropdown("Collapse Depth", button_content="icon-sort-number-up",
                            uid="idCollapseDepthDiv")

        group3 = toolbar.add_button_group()
        group3.add_button("Clear Arrows and Connections", uid="clearArrowsAndConnectsButtonId",
                          content="icon-eraser")
        group3.add_button("Show Path", uid="showCurrentPathButtonId", content="icon-terminal")
        group3.add_button("Show Legend", uid="showLegendButtonId", content="icon-map-signs")
        group3.add_button("Show Params", uid="showParamsButtonId", content="icon-exchange")
        group3.add_button("Toggle Solver Names", uid="toggleSolverNamesButtonId",
                          content="icon-minus")
        group3.add_dropdown("Font Size", id_naming="idFontSize", options=_FONT_SIZES,
                            option_formatter=lambda x: '{}px'.format(x),
                            button_content="icon-text-height")
        group3.add_dropdown("Vertically Resize", id_naming="idVerticalResize",
                            options=_MODEL_HEIGHTS, option_formatter=lambda x: '{}px'.format(x),
                            button_content="icon-resize-vertical", header="Model Height")

        group4 = toolbar.add_button_group()
        group4.add_button("Save SVG", uid="saveSvgButtonId", content="icon-floppy")

        group5 = toolbar.add_button_group()
        group5.add_button("Help", uid="helpButtonId", content="icon-help")

        # Help
        help_txt = ('XDSM help.')

        h.add_help(help_txt, footer="OpenMDAO XDSM diagram")

        h.insert("{{magic}}", _HTML_TMP)
        if toolbar:
            h.insert("{{scripts}}", scripts)
            h.insert('{{fontello}}', encoded_font)

        # Write output file
        h.write(outfile)


if __name__ == '__main__':
    # with JSON file name as input
    write_html(outfile='xdsmjs/xdsm_diagram.html', source_data="examples/idf.json")

    # with JSON data as input
    with open("XDSMjs/examples/idf.json") as f:
        data = json.load(f)
    write_html(outfile='xdsm_diagram_data_embedded.html', source_data=data)
