IN_PROGRESS = "In Progress"
DONE = "Done"
DONE_OV = "Done (Overriden)"
NOT_DONE = "Not Done"

status_color = {
    DONE: "#5D7B5B",
    DONE_OV: "#8F4D4D",
    IN_PROGRESS: "#BEA566",
    NOT_DONE: "#414141",
}

status_text = {
    DONE: " \u2713 ",
    DONE_OV: " \u2713 ",
    IN_PROGRESS: " IP ",
    NOT_DONE: "",
}

colors = {
    DONE: {
        'bg': "#1F1F1F",
        'txt': "#747474"},
    DONE_OV: {
        'bg': "#1F1F1F",
        'txt': "#747474"},
    IN_PROGRESS: {
        'bg': "#C8C8C8",
        'txt': "#474747"},
    NOT_DONE: {
        'bg': "#414141",
        'txt': "#A1A1A1"},
}

dark_theme_background = "#252525"
override_button_color = "#932323"
live_stream_url = "https://api.ucsb.edu/dining/cams/v2/stream/carrillo?ucsb-api-key=0AVpBy0HfloWaEQHPanHTGSYmXusaNIJ"
# live_stream_url = "http://d3rlna7iyyu8wu.cloudfront.net/DolbyVision_Atmos/profile8.1_DASH/p8.1.mpd"
