import rospy


class CallbackWrapper():
    def __init__(self, publishers_meta) -> None:
        super().__init__()
        self.publishers = None
        self.callback = None
        self.publishers_meta = publishers_meta

    def get_publishers(self):
        if self.publishers is None:
            self.publishers = {entry['name']:
                                   rospy.Publisher(entry['topic'],
                                                   entry['message'],
                                                   **entry.get('kwargs', {}))
                               for entry in self.publishers_meta}
        return self.publishers

    def register_callback(self, callback):
        self.callback = callback

    def service_callback(self, request):
        return self.callback(request, self.get_publishers())
