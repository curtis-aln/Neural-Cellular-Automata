from colorama import Fore, Back, Style, init

init()

# Define a dictionary for foreground colors
foreground_colors = {
    'black': Fore.BLACK,
    'red': Fore.RED,
    'green': Fore.GREEN,
    'yellow': Fore.YELLOW,
    'blue': Fore.BLUE,
    'magenta': Fore.MAGENTA,
    'cyan': Fore.CYAN,
    'white': Fore.WHITE,
    'reset': Fore.RESET,
}

# Define a dictionary for background colors
background_colors = {
    'black': Back.BLACK,
    'red': Back.RED,
    'green': Back.GREEN,
    'yellow': Back.YELLOW,
    'blue': Back.BLUE,
    'magenta': Back.MAGENTA,
    'cyan': Back.CYAN,
    'white': Back.WHITE,
    'reset': Back.RESET
}


def print_col(text, fg_color='reset', bg_color=None) -> None:
    """
    Print text in specified foreground and background color.

    Parameters
    ----------
        text (str): text to print
        fg_color (str): foreground color
        bg_color (str): background color

    Colors
    -----
    - black
    - red
    - green
    - yellow
    - blue
    - magenta
    - cyan
    - white
    - reset
    """
    fg = foreground_colors.get(fg_color.lower(), Fore.RESET)
    bg = background_colors.get(bg_color.lower(), '') if bg_color else ''
    
    print(f"{fg}{bg}{text}{Style.RESET_ALL}")