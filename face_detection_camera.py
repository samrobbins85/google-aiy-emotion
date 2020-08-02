import argparse

from picamera import PiCamera

from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from aiy.vision.annotator import Annotator
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from environs import Env
import urllib.request, json
def avg_joy_score(faces):
    if faces:
        return sum(face.joy_score for face in faces) / len(faces)
    return 0.0


def main():
    env = Env()
    env.read_env()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', '-n', type=int, dest='num_frames', default=None,
                        help='Sets the number of frames to run for, otherwise runs forever.')
    args = parser.parse_args()

    # Forced sensor mode, 1640x1232, full FoV. See:
    # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
    # This is the resolution inference run on.
    with PiCamera(sensor_mode=4, resolution=(1640, 1232), framerate=30) as camera:
        camera.start_preview()

        # Annotator renders in software so use a smaller size and scale results
        # for increased performace.
        annotator = Annotator(camera, dimensions=(320, 240))
        scale_x = 320 / 1640
        scale_y = 240 / 1232

        # Incoming boxes are of the form (x, y, width, height). Scale and
        # transform to the form (x1, y1, x2, y2).
        def transform(bounding_box):
            x, y, width, height = bounding_box
            return (scale_x * x, scale_y * y, scale_x * (x + width),
                    scale_y * (y + height))

        joy_counter = 0
        with CameraInference(face_detection.model()) as inference:
            for result in inference.run(args.num_frames):
                faces = face_detection.get_faces(result)
                annotator.clear()
                for face in faces:
                    annotator.bounding_box(transform(face.bounding_box), fill=0)
                annotator.update()
                if len(faces) > 0:
                    if avg_joy_score(faces) > 0.8:
                        if joy_counter < 0:
                            joy_counter = 0
                        else:
                            joy_counter += 1

                    if avg_joy_score(faces) < 0.1:
                        if joy_counter > 0:
                            joy_counter = 0
                        else:
                            joy_counter -= 1
                    if joy_counter > 20:
                        print("Happy")


                        joy_counter = 0
                    if joy_counter < -20:
                        print("Sad")
                        request_url = urllib.request.urlopen("https://www.reddit.com/r/dogpictures/random.json")
                        result = json.loads(request_url.read().decode())[0]["data"]["children"][0]["data"]["url"]
                        message = Mail(
                            from_email='contact@samrobbins.uk',
                            to_emails='samrobbinsgb@gmail.com',
                            subject='Sending with Twilio SendGrid is Fun',
                            html_content='<img src='+result+'>')
                        try:
                            sg = SendGridAPIClient(env.str('SENDGRID_API_KEY'))
                            response = sg.send(message)
                            print(response.status_code)
                            # print(response.body)
                            # print(response.headers)
                        except Exception as e:
                            print(e.message)
                        joy_counter = 0
                else:
                    joy_counter = 0
                    # print('#%05d (%5.2f fps): num_faces=%d, avg_joy_score=%.2f' %
                #     (inference.count, inference.rate, len(faces), avg_joy_score(faces)))

        camera.stop_preview()


if __name__ == '__main__':
    main()
