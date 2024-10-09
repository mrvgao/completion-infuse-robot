prompt_configs = {
  "with_complete_rate": """these three images are left, mid, and right images of a robot who is executing the task {task} The current task completion rate is {complete_rate}% . Please specify what state's the robot is, actions of the robot here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please give what the robot should do to avoid the error. Make sure the answer is clear, specific and less than 100 words. Don't repeat any information that is already given in the question.
                          Answer the question as the following format:
                          State:
                          Next Action:
                          Potential Error:
                          Error Avoidance:""",
  "without_complete_rate": """these three images are left, mid, and right images of a robot who is executing the task {task}.Please specify what state's the robot is, actions of the robot here should be taken next, and what the errors will be probably made by the robot.And if there is any potential error, please give what the robot should do to avoid the error. Make sure the answer is clear, specific and less than 100 words. Don't repeat any information that is already given in the question.
                          Answer the question as the following format:
                          State:
                          Next Action:
                          Potential Error:
                          Error Avoidance:"""
}

