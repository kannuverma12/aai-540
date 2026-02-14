import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.workflow.pipeline_context import PipelineSession

session = PipelineSession()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()
region = session.boto_region_name

# --------------------------
# Parameters
# --------------------------
input_data = ParameterString(
    name="InputDataUrl",
    default_value=f"s3://{bucket}/credit-risk/raw/data.csv"
)

accuracy_threshold = ParameterFloat(
    name="AccuracyThreshold",
    default_value=0.75
)

# --------------------------
# Processing Step
# --------------------------
processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, "1.0-1"),
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=role
)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/train",
            destination=f"s3://{bucket}/credit-risk/processed/train"
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/test",
            destination=f"s3://{bucket}/credit-risk/processed/test"
        )
    ],
    code="scripts/preprocess.py"
)

# --------------------------
# Training Step
# --------------------------
estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", region, "1.5-1"),
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{bucket}/credit-risk/models"
)

estimator.set_hyperparameters(
    objective="binary:logistic",
    num_round=200,
    max_depth=5,
    eta=0.2
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": processing_step.properties.ProcessingOutputConfig.Outputs[
            "train"
        ].S3Output.S3Uri
    }
)

# --------------------------
# Evaluation Step
# --------------------------
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{bucket}/credit-risk/evaluation"
        )
    ],
    code="scripts/evaluate.py",
    property_files=[evaluation_report]
)

# --------------------------
# Conditional Step
# --------------------------
cond_step = ConditionStep(
    name="AccuracyCheck",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=evaluation_report.prop("metrics.accuracy"),
            right=accuracy_threshold
        )
    ],
    if_steps=[],
    else_steps=[]
)

# --------------------------
# Register Model
# --------------------------
register_step = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_package_group_name="CreditRiskModelGroup",
    approval_status="PendingManualApproval"
)

# --------------------------
# Pipeline
# --------------------------
pipeline = Pipeline(
    name="CreditRiskPipeline",
    parameters=[input_data, accuracy_threshold],
    steps=[
        processing_step,
        training_step,
        evaluation_step,
        cond_step,
        register_step
    ]
)

pipeline.upsert(role_arn=role)
pipeline.start()
