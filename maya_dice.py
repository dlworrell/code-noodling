import maya.cmds as cmds
import math

def create_dice(sides, size=1.0):
    """
    Create a die with a specified number of sides and carve numbers into the center of each face.
    
    Args:
        sides (int): Number of sides on the die (supports 4, 6, 8, 12, 20, and up to 100).
        size (float): Diameter of the die.
    """
    # Supported Platonic solids or approximate sphere generation
    if sides in [4, 6, 8, 12, 20]:  # Platonic solids
        die = create_platonic_die(sides, size)
    elif sides <= 100:  # Approximate with sphere-like polyhedron
        die = create_geodesic_die(sides, size)
    else:
        raise ValueError(f"Sides {sides} not supported. Must be 4-100.")

    # Add numbers to the die faces
    add_numbers_to_faces(die, sides, size)

    # Export the die as FBX
    cmds.select(die)
    cmds.file(rename=f"d{sides}_die_with_numbers.fbx")
    cmds.file(exportSelected=True, type="FBX export")
    print(f"Die with {sides} sides exported as FBX.")

    return die


def create_platonic_die(sides, size):
    """
    Create a platonic solid die (4, 6, 8, 12, 20 sides).
    """
    if sides == 4:
        die = cmds.polyPlatonicSolid(solidType=1, radius=size / 2, name="d4_die")[0]  # Tetrahedron
    elif sides == 6:
        die = cmds.polyCube(w=size, h=size, d=size, name="d6_die")[0]  # Cube
    elif sides == 8:
        die = cmds.polyPlatonicSolid(solidType=2, radius=size / 2, name="d8_die")[0]  # Octahedron
    elif sides == 12:
        die = cmds.polyPlatonicSolid(solidType=4, radius=size / 2, name="d12_die")[0]  # Dodecahedron
    elif sides == 20:
        die = cmds.polyPlatonicSolid(solidType=5, radius=size / 2, name="d20_die")[0]  # Icosahedron
    else:
        raise ValueError("Unsupported platonic die.")
    return die


def create_geodesic_die(sides, size):
    """
    Create an approximate die with a high number of sides using geodesic tessellation.
    """
    # Create a sphere and tessellate it to approximate the desired number of faces
    die = cmds.polySphere(radius=size / 2, subdivisionsAxis=int(math.sqrt(sides) * 2),
                          subdivisionsHeight=int(math.sqrt(sides) * 2), name=f"d{sides}_die")[0]
    cmds.polyTriangulate(die)  # Ensure all faces are triangles
    cmds.polyReduce(die, ver=0.5, vertexCount=sides, keepQuadsWeight=0.0)  # Reduce faces to target number
    return die


def add_numbers_to_faces(die, sides, size):
    """
    Add carved numbers to each face of the die.
    """
    # Get the center of each face
    face_centers = cmds.polyEvaluate(die, face=True)
    if face_centers < sides:
        print("Warning: Unable to generate sufficient faces for numbering.")
    for i in range(sides):
        # Create text for the face number
        text = cmds.textCurves(ch=False, f="Arial|w400|h100", t=str(i + 1))[0]
        cmds.group(text, name=f"face_{i + 1}_text")
        
        # Find the face center and normal
        face_center = cmds.polyInfo(die + f".f[{i}]", faceNormals=True)
        center_pos = extract_face_center(face_center)
        
        # Position and scale the number
        cmds.xform(text, s=[0.1, 0.1, 0.1])  # Scale down
        cmds.xform(text, t=center_pos)  # Translate to face center

        # Carve the number into the die
        cmds.polyBoolOp(die, text, op=2)  # Union operation


def extract_face_center(face_data):
    """
    Extract the face center and normal vector from face data.
    """
    face_split = face_data.split()
    center_pos = [float(face_split[i]) for i in range(2, 5)]  # Extract x, y, z
    return center_pos


# Example: Create dice with up to 100 sides
for sides in [4, 6, 8, 12, 20, 30, 50, 100]:
    create_dice(sides)