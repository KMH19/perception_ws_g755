# Import the necessary message types
from obr_msgs.msg import Cone2
from geometry_msgs.msg import Transform, Point, Vector3, Quaternion

# Create a Cone2 message
cone2_msg = Cone2(
    position=Point(x=1.4742001295089722, y=-0.043905806354907026, z=0.0),
    label=Label(label=1),
    confidence=1.0
)

# Create a Transform message
transform_msg = Transform(
    translation=Vector3(x=-0.010261125960951705, y=0.06037561433184098, z=0.015499791692706783),
    rotation=Quaternion(x=1.6328222195786408e-05, y=0.024997390581939498, z=-0.0006529928136432988, w=0.9996873030092301)
)

# Apply the transformation to the Cone2 message
transformed_cone2_msg = Cone2(
    position=Point(
        x=cone2_msg.position.x + transform_msg.translation.x,
        y=cone2_msg.position.y + transform_msg.translation.y,
        z=cone2_msg.position.z + transform_msg.translation.z
    ),
    label=cone2_msg.label,
    confidence=cone2_msg.confidence
)

# Print the transformed Cone2 message
print(transformed_cone2_msg)