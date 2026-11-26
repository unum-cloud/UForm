// The training verbs: a mesh joining a run, a trainer, and its receivers.
//
// File: golang/train.go
// Author: Ash Vardanian
// Date: November 18, 2026

package uform

import "unsafe"

/*
// Bootstrap, describe and step take no callback; init and save report through a progress handler.
#cgo noescape uform_mesh_bootstrap
#cgo nocallback uform_mesh_bootstrap
#cgo noescape uform_mesh_describe
#cgo nocallback uform_mesh_describe
#cgo noescape uform_trainer_init_vram
#cgo noescape uform_trainer_step
#cgo nocallback uform_trainer_step
#cgo noescape uform_trainer_save
#cgo noescape uform_modality_from_name
#cgo nocallback uform_modality_from_name
#cgo noescape uform_objective_from_name
#cgo nocallback uform_objective_from_name
#cgo noescape uform_negatives_from_name
#cgo nocallback uform_negatives_from_name
#cgo noescape uform_pair_loss_form_from_name
#cgo nocallback uform_pair_loss_form_from_name
#cgo noescape uform_late_interaction_side_from_name
#cgo nocallback uform_late_interaction_side_from_name
#cgo noescape uform_multimem_tier_from_name
#cgo nocallback uform_multimem_tier_from_name

#include "uform/train.h"
*/
import "C"

// region: Enums

// Modality is what the bytes of one training item hold.
type Modality int32

const (
	// ModalityText is UTF-8 text, a byte per token.
	ModalityText Modality = C.uform_modality_text_k
	// ModalityImage is a JPEG, PNG or WebP file.
	ModalityImage Modality = C.uform_modality_image_k
	// ModalityVideo is an MP4 or WebM file.
	ModalityVideo Modality = C.uform_modality_video_k
	// ModalityAudio is a WAV, FLAC, MP3 or Opus file.
	ModalityAudio Modality = C.uform_modality_audio_k
)

// String renders the modality as the library spells it.
func (modality Modality) String() string {
	return C.GoString(C.uform_modality_name(C.uform_modality_t(modality)))
}

// ModalityFromName reads a modality back from the spelling [Modality.String] renders.
func ModalityFromName(name string) (Modality, error) {
	var modality C.uform_modality_t
	err := named(C.uform_modality_from_name(view(name), &modality))
	return Modality(modality), err
}

// Objective is which losses a step relaxes, one per combination the engine trains.
type Objective int32

const (
	// ObjectiveMasked is masked reconstruction alone, as the Base pretrains.
	ObjectiveMasked Objective = C.uform_objective_masked_k
	// ObjectiveContrastive is the pairwise contrastive loss alone, as the Embedder trains.
	ObjectiveContrastive Objective = C.uform_objective_contrastive_k
	// ObjectiveJoint is both, joined at the final-norm states.
	ObjectiveJoint Objective = C.uform_objective_joint_k
)

// String renders the objective as the library spells it.
func (objective Objective) String() string {
	return C.GoString(C.uform_objective_name(C.uform_objective_t(objective)))
}

// ObjectiveFromName reads an objective back from the spelling [Objective.String] renders.
func ObjectiveFromName(name string) (Objective, error) {
	var objective C.uform_objective_t
	err := named(C.uform_objective_from_name(view(name), &objective))
	return Objective(objective), err
}

// Negatives is which pairs the contrastive loss pushes apart.
type Negatives int32

const (
	// NegativesAmbient repels edges, and every pair no edge names at the ambient weight.
	NegativesAmbient Negatives = C.uform_negatives_ambient_k
	// NegativesNamed repels edges alone.
	NegativesNamed Negatives = C.uform_negatives_named_k
)

// String renders the policy as the library spells it.
func (negatives Negatives) String() string {
	return C.GoString(C.uform_negatives_name(C.uform_negatives_t(negatives)))
}

// NegativesFromName reads a negatives policy back from the spelling [Negatives.String] renders.
func NegativesFromName(name string) (Negatives, error) {
	var negatives C.uform_negatives_t
	err := named(C.uform_negatives_from_name(view(name), &negatives))
	return Negatives(negatives), err
}

// PairLossForm is how a contrastive term turns a row of scores into forces.
type PairLossForm int32

const (
	// PairLossFormDerived keeps the term's own form: sigmoid for pooled vertices, ranked for tokens.
	PairLossFormDerived PairLossForm = C.uform_pair_loss_form_derived_k
	// PairLossFormBalanced normalizes the attract and repel halves of every cell apart.
	PairLossFormBalanced PairLossForm = C.uform_pair_loss_form_balanced_k
	// PairLossFormRanked scores every positive over every negative, where the bias cancels.
	PairLossFormRanked PairLossForm = C.uform_pair_loss_form_ranked_k
	// PairLossFormSigmoid is SigLIP's per-cell sigmoid cross-entropy.
	PairLossFormSigmoid PairLossForm = C.uform_pair_loss_form_sigmoid_k
)

// String renders the form as the library spells it.
func (form PairLossForm) String() string {
	return C.GoString(C.uform_pair_loss_form_name(C.uform_pair_loss_form_t(form)))
}

// PairLossFormFromName reads a pair loss form back from the spelling [PairLossForm.String] renders.
func PairLossFormFromName(name string) (PairLossForm, error) {
	var form C.uform_pair_loss_form_t
	err := named(C.uform_pair_loss_form_from_name(view(name), &form))
	return PairLossForm(form), err
}

// LateInteractionSide is which side of a late-interaction score an item takes.
type LateInteractionSide int32

const (
	// LateInteractionSideNone takes no part in late interaction.
	LateInteractionSideNone LateInteractionSide = C.uform_late_interaction_side_none_k
	// LateInteractionSideQuery scores the documents the item attracts and repels.
	LateInteractionSideQuery LateInteractionSide = C.uform_late_interaction_side_query_k
	// LateInteractionSideDocument is scored by the queries naming it.
	LateInteractionSideDocument LateInteractionSide = C.uform_late_interaction_side_document_k
	// LateInteractionSideQueryAndDocument plays both sides.
	LateInteractionSideQueryAndDocument LateInteractionSide = C.uform_late_interaction_side_query_and_document_k
)

// String renders the side as the library spells it.
func (side LateInteractionSide) String() string {
	return C.GoString(C.uform_late_interaction_side_name(C.uform_late_interaction_side_t(side)))
}

// LateInteractionSideFromName reads a side back from the spelling [LateInteractionSide.String]
// renders.
func LateInteractionSideFromName(name string) (LateInteractionSide, error) {
	var side C.uform_late_interaction_side_t
	err := named(C.uform_late_interaction_side_from_name(view(name), &side))
	return LateInteractionSide(side), err
}

// MultimemTier is how a mesh's NVLink domains reduce.
type MultimemTier int32

const (
	// MultimemTierSwitchReduced reduces through the NVLink switch's multicast memory.
	MultimemTierSwitchReduced MultimemTier = C.uform_multimem_tier_switch_reduced_k
	// MultimemTierPeerRead reads every peer in a fixed order, where multicast is absent.
	MultimemTierPeerRead MultimemTier = C.uform_multimem_tier_peer_read_k
)

// String renders the tier as the library spells it.
func (tier MultimemTier) String() string {
	return C.GoString(C.uform_multimem_tier_name(C.uform_multimem_tier_t(tier)))
}

// MultimemTierFromName reads a multimem tier back from the spelling [MultimemTier.String] renders.
func MultimemTierFromName(name string) (MultimemTier, error) {
	var tier C.uform_multimem_tier_t
	err := named(C.uform_multimem_tier_from_name(view(name), &tier))
	return MultimemTier(tier), err
}

// endregion: Enums

// region: Configs

// ImageShape is the image slice of a batch envelope; a zero PatchSide leaves images out of the run.
//
// Mirrors uform_image_shape_t.
type ImageShape struct {
	// PatchSide is the square patch side images tile into, in pixels.
	PatchSide uint
	// Channels is the channels every pixel carries; zero derives four.
	Channels uint
}

// VideoShape is the video slice of a batch envelope; a zero PatchSide leaves video out of the run.
//
// Mirrors uform_video_shape_t.
type VideoShape struct {
	// PatchSide is the square patch side one tube tiles into, in pixels.
	PatchSide uint
	// TubeFrames is the frames one tube folds ahead of its patch; zero derives two.
	TubeFrames uint
	// FrameStride is the source frames between sampled ones; zero derives one.
	FrameStride uint
	// Channels is the channels every pixel carries; zero derives three.
	Channels uint
}

// AudioShape is the audio slice of a batch envelope; a zero ChunkSamples leaves audio out of the
// run.
//
// Mirrors uform_audio_shape_t.
type AudioShape struct {
	// ChunkSamples is the raw samples per frame.
	ChunkSamples uint
	// SampleRate is the mono rate every recording is resampled to; zero derives 16000.
	SampleRate uint
}

// BatchShape is what one device's step carries at most, whatever the modality mix.
//
// Mirrors uform_batch_shape_t field for field.
type BatchShape struct {
	// MaxRowsPerDevice is the rows one device's step packs.
	MaxRowsPerDevice uint
	// MaxSegmentsPerDevice is the items one device's step carries across all its rows.
	MaxSegmentsPerDevice uint
	// MaxColumnsPerRow is the positions one row spans.
	MaxColumnsPerRow uint
	// MaxBagsPerDevice is the bags one device's step carries; zero keeps every item its own unit.
	MaxBagsPerDevice uint
	// MaxChoicesPerItem is the legal bytes one item offers at most; zero makes no choices.
	MaxChoicesPerItem uint
	// Image is the image geometry.
	Image ImageShape
	// Video is the video geometry.
	Video VideoShape
	// Audio is the audio geometry.
	Audio AudioShape
}

// Masking is how the masked objective corrupts a batch and weighs its heads; every zero derives.
//
// Mirrors uform_masking_t field for field.
type Masking struct {
	TokenMaskProbability float32
	TokenMeanSpan        float32
	ReplaceFraction      float32
	RandomFraction       float32
	ImageMaskRatio       float32
	VideoMaskRatio       float32
	AudioMaskRatio       float32
	TextWeight           float32
	ImageWeight          float32
	VideoWeight          float32
	AudioWeight          float32
}

// Contrastive is how the contrastive loss scores the pooled embeddings; every zero derives.
//
// Mirrors uform_contrastive_t.
type Contrastive struct {
	// MatryoshkaDimensions are the prefix widths scored, strictly ascending multiples of 8, at most
	// 8 of them; empty scores the full width alone.
	MatryoshkaDimensions []uint
	// TemperatureGroups is the learnable pairs of an inverse temperature and a bias; zero derives
	// one.
	TemperatureGroups uint
	// InverseTemperature is the starting inverse temperature of every group.
	InverseTemperature float32
	// Bias is the starting logit bias of every group.
	Bias float32
	// Negatives is which pairs repel.
	Negatives Negatives
	// AmbientRepulsionWeight is the weight of every pair no edge names under NegativesAmbient.
	AmbientRepulsionWeight float32
	// MaxEdgesPerDevice is the edges one device's step names per edge set at most; zero derives.
	MaxEdgesPerDevice uint
	// PairLoss is the pair loss over each anchor's row.
	PairLoss PairLossForm
}

// LateInteraction is how the late-interaction loss scores per-token vectors; every zero derives.
//
// Mirrors uform_late_interaction_t field for field.
type LateInteraction struct {
	Weight                     float32
	Width                      uint
	InverseTemperature         float32
	Bias                       float32
	MaxQuerySpan               uint
	MaxQueryTokensPerDevice    uint
	MaxDocumentsPerDevice      uint
	MaxDocumentTokensPerDevice uint
	PairLoss                   PairLossForm
}

// Choices is the calibrated choices term over the byte decoder; a zero Weight trains none.
//
// Mirrors uform_choices_t field for field.
type Choices struct {
	Weight      float32
	BrierWeight float32
}

// Schedule is the learning-rate schedule: a linear warmup to the peak, then a cosine decay.
//
// Mirrors uform_schedule_t field for field.
type Schedule struct {
	LearningRate    float32
	MinLearningRate float32
	WarmupSteps     uint
	Steps           uint
}

// MeshOptions is where this process sits among the processes of a run, and how they meet.
//
// Mirrors uform_mesh_options_t.
type MeshOptions struct {
	// HostDomainIndex is this process's index among HostDomainCount.
	HostDomainIndex uint
	// HostDomainCount is the processes in the run; zero derives one.
	HostDomainCount uint
	// GPUsPerMultimemDomain is the GPUs one NVLink multicast domain spans; zero derives the host.
	GPUsPerMultimemDomain uint
	// Directory is the path every host sees, where processes exchange their facts.
	Directory string
	// Job is the run's unique name, naming one directory inside Directory.
	Job string
	// TimeoutSeconds is how long any wait for a peer lasts; zero derives 120.
	TimeoutSeconds float64
}

// MeshDescription is what a mesh reports about itself.
//
// Mirrors uform_mesh_description_t field for field, so the library writes it in place.
type MeshDescription struct {
	GPUsPerHostDomain      uint
	GPUsPerMultimemDomain  uint
	GPUsPerColocatedDomain uint
	HostDomainIndex        uint
	FirstGPUIndex          uint
	MultimemTier           MultimemTier
	GPUClass               GPUClass
}

// TrainerInitVramConfig is the geometry, objectives and schedule of a run; zero fields derive.
//
// Mirrors uform_trainer_init_vram_t.
type TrainerInitVramConfig struct {
	// Resume is the path of a checkpoint to continue from; empty starts from Seed.
	Resume string
	// Depth is the transformer blocks stacked.
	Depth uint
	// Hidden is the embedding width, a multiple of 128.
	Hidden uint
	// Batch is what one device's step carries at most.
	Batch BatchShape
	// Objective is which losses a step relaxes.
	Objective Objective
	// Masking configures the masked objective.
	Masking Masking
	// Contrastive configures the contrastive loss.
	Contrastive Contrastive
	// LateInteraction configures the late-interaction loss.
	LateInteraction LateInteraction
	// Choices configures the calibrated choices term.
	Choices Choices
	// Schedule is the learning-rate schedule.
	Schedule Schedule
	// Mesh must stay open until the trainer is closed; nil runs this process alone.
	Mesh *Mesh
	// Seed is the base of every keyed draw.
	Seed uint64
	// Progress hears the tensors placed on a resume.
	Progress ProgressCallback
	// ProgressSteps is the reports the call makes at most; zero derives ten.
	ProgressSteps uint
}

// Edges is one edge set over a step's items as compressed sparse rows: item i links to
// IDs[Offsets[i]] up to IDs[Offsets[i+1]].
//
// Mirrors uform_edges_view_t.
type Edges struct {
	// Offsets holds one more entry than the step's items; empty names no edges.
	Offsets []uint
	// IDs are indices into the step's items, ascending within each run.
	IDs []uint32
	// Weights holds one weight per ID; empty weighs every edge one.
	Weights []float32
}

// Relations is the contrastive targets of one step.
//
// Mirrors uform_relations_view_t.
type Relations struct {
	Attract Edges
	Repel   Edges
	Ignore  Edges
}

// TrainerStepStats is what a step reports, every loss its world mean.
//
// Mirrors uform_trainer_step_stats_t field for field, so the library writes it in place.
type TrainerStepStats struct {
	Step                uint
	LearningRate        float32
	Loss                float32
	ContrastiveLoss     float32
	MaskedTextLoss      float32
	MaskedImageLoss     float32
	MaskedVideoLoss     float32
	MaskedAudioLoss     float32
	LateInteractionLoss float32
	ChoicesLoss         float32
	Truncated           uint
	Dropped             uint
}

// TrainerStepConfig holds the per-item inputs and knobs of a step beside its modalities and
// contents; nil derives every field.
//
// Mirrors uform_trainer_step_t.
type TrainerStepConfig struct {
	// Bags holds the bag of each item; empty keeps every item its own bag.
	Bags []uint32
	// TemperatureGroups holds the temperature group of each item; empty puts every item in group zero.
	TemperatureGroups []uint32
	// Sides holds the late-interaction side of each item; empty takes none into late interaction.
	Sides []LateInteractionSide
	// Choices holds each item's legal bytes, as TextChoose takes them; the zero Tape makes none.
	Choices Tape
	// ChoiceTargets holds the target probability of each choice, as TextChoose answers them.
	ChoiceTargets Tape
	// Relations holds the attract, repel and ignore edges.
	Relations Relations
	// Arena receives the pooled embeddings the loss scored; nil skips the copy.
	Arena *Arena
	// OutputsScalar is the scalar the embeddings land as.
	OutputsScalar Scalar
	// Stats is the report the step overwrites, kept by the caller until it is final; nil skips it.
	Stats *TrainerStepStats
}

// TrainerSaveConfig holds the knobs of Save; nil derives every field.
type TrainerSaveConfig struct {
	// Progress hears the tensors written.
	Progress ProgressCallback
	// ProgressSteps is the reports the call makes at most; zero derives ten.
	ProgressSteps uint
}

// Every struct the library reads or writes in place matches its C twin byte for byte.
var (
	_ = [1]byte{}[unsafe.Sizeof(BatchShape{})-C.sizeof_uform_batch_shape_t]
	_ = [1]byte{}[unsafe.Sizeof(Masking{})-C.sizeof_uform_masking_t]
	_ = [1]byte{}[unsafe.Sizeof(LateInteraction{})-C.sizeof_uform_late_interaction_t]
	_ = [1]byte{}[unsafe.Sizeof(Choices{})-C.sizeof_uform_choices_t]
	_ = [1]byte{}[unsafe.Sizeof(Schedule{})-C.sizeof_uform_schedule_t]
	_ = [1]byte{}[unsafe.Sizeof(MeshDescription{})-C.sizeof_uform_mesh_description_t]
	_ = [1]byte{}[unsafe.Sizeof(TrainerStepStats{})-C.sizeof_uform_trainer_step_stats_t]
)

// check refuses what only the binding sees of an edge set: offsets for another count than the
// step's, or weights not paired with the IDs.
func (edges Edges) check(count uint) error {
	if len(edges.Offsets) != 0 && uint(len(edges.Offsets)) != count+1 {
		return fail(StatusShapeMismatch, "an edge set holds one more offset than the step's items")
	}
	if len(edges.Weights) != 0 && len(edges.Weights) != len(edges.IDs) {
		return fail(StatusShapeMismatch, "pass no weights or one per edge")
	}
	return nil
}

// edges pins one edge set until the call is final.
func (call *call) edges(edges Edges) C.uform_edges_view_t {
	return C.uform_edges_view_t{
		offsets: (*C.size_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(edges.Offsets)), len(edges.Offsets))),
		ids:     call.uint32s(edges.IDs),
		weights: (*C.float)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(edges.Weights)), len(edges.Weights))),
	}
}

// endregion: Configs

// region: Mesh

// Mesh is one process's place among the processes of a run, made once and referred to by every
// trainer, which must be closed first.
//
// Mirrors uform_mesh_t; Close calls uform_mesh_free.
type Mesh struct {
	handle C.uform_mesh_t
}

// MeshBootstrap joins this process to the run's mesh over executor, a VRAM executor whose devices
// are this process's members; every process of the run calls it with the same options.
//
// Calls uform_mesh_bootstrap, blocking.
func MeshBootstrap(executor *Executor, config *MeshOptions) (*Mesh, error) {
	options := configOf(config)
	handle, err := executor.lower()
	if err != nil {
		return nil, err
	}
	var call callWithResult[C.uform_mesh_t]
	arguments := open[C.uform_mesh_bootstrap_t](&call)
	*arguments = C.uform_mesh_bootstrap_t{
		mesh:     call.result,
		executor: handle,
		outcome:  call.outcome(),
		options: C.uform_mesh_options_t{
			host_domain_index:        C.size_t(options.HostDomainIndex),
			host_domain_count:        C.size_t(options.HostDomainCount),
			gpus_per_multimem_domain: C.size_t(options.GPUsPerMultimemDomain),
			directory:                call.text(options.Directory),
			job:                      call.text(options.Job),
			timeout_seconds:          C.double(options.TimeoutSeconds),
		},
	}
	C.uform_mesh_bootstrap(arguments)
	mesh, err := block(&call, nil)
	if err != nil {
		return nil, err
	}
	return &Mesh{handle: mesh}, nil
}

// Describe reports the mesh's shape and this process's place in it.
//
// Calls uform_mesh_describe.
func (mesh *Mesh) Describe() (MeshDescription, error) {
	if mesh == nil || mesh.handle == nil {
		return MeshDescription{}, fail(StatusUninitialized, "the mesh is closed or was never joined")
	}
	var call callWithResult[MeshDescription]
	arguments := open[C.uform_mesh_describe_t](&call)
	*arguments = C.uform_mesh_describe_t{
		mesh:        mesh.handle,
		outcome:     call.outcome(),
		description: (*C.uform_mesh_description_t)(unsafe.Pointer(call.result)),
	}
	C.uform_mesh_describe(arguments)
	return block(&call, nil)
}

// Close leaves the mesh. It is idempotent, and every trainer made over the mesh must be closed
// first.
func (mesh *Mesh) Close() error {
	if mesh == nil || mesh.handle == nil {
		return nil
	}
	C.uform_mesh_free(mesh.handle)
	mesh.handle = nil
	return nil
}

// endregion: Mesh

// region: Trainer

// Trainer is a training run: the tower replicas, gradients, optimizer moments, heads and
// temperatures on its executor's devices, stepped and saved through that executor.
//
// Its embedded TrainerOn runs every verb on the calling goroutine, and TrainerOn.On on the executor
// the trainer was made on.
//
// Mirrors uform_trainer_t; Close calls uform_trainer_free.
type Trainer struct {
	TrainerOn
	handle C.uform_trainer_t
}

// trainerOf wraps the handle an init wrote.
func trainerOf(handle C.uform_trainer_t, err error) (*Trainer, error) {
	if err != nil {
		return nil, err
	}
	trainer := &Trainer{handle: handle}
	trainer.TrainerOn = TrainerOn{trainerReceiver{receiver[C.uform_trainer_t]{handle: &trainer.handle}}}
	return trainer, nil
}

// TrainerInitVram builds one replica per device of executor and starts or resumes the run; every
// process of the run calls it with the same geometry, objective and seed.
//
// Calls uform_trainer_init_vram, blocking.
func TrainerInitVram(executor *Executor, config *TrainerInitVramConfig) (*Trainer, error) {
	options := configOf(config)
	handle, err := executor.lower()
	if err != nil {
		return nil, err
	}
	var mesh C.uform_mesh_t
	if options.Mesh != nil {
		mesh = options.Mesh.handle
	}
	var call callWithResult[C.uform_trainer_t]
	arguments := open[C.uform_trainer_init_vram_t](&call)
	contrastive := options.Contrastive
	*arguments = C.uform_trainer_init_vram_t{
		trainer:          call.result,
		executor:         handle,
		outcome:          call.outcome(),
		resume:           call.text(options.Resume),
		depth:            C.size_t(options.Depth),
		hidden:           C.size_t(options.Hidden),
		batch:            *(*C.uform_batch_shape_t)(unsafe.Pointer(&options.Batch)),
		objective:        C.uform_objective_t(options.Objective),
		masking:          *(*C.uform_masking_t)(unsafe.Pointer(&options.Masking)),
		late_interaction: *(*C.uform_late_interaction_t)(unsafe.Pointer(&options.LateInteraction)),
		choices:          *(*C.uform_choices_t)(unsafe.Pointer(&options.Choices)),
		schedule:         *(*C.uform_schedule_t)(unsafe.Pointer(&options.Schedule)),
		mesh:             mesh,
		seed:             C.uint64_t(options.Seed),
		progress:         call.progress(options.Progress, options.ProgressSteps),
		contrastive: C.uform_contrastive_t{
			matryoshka_dimensions:       (*C.size_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(contrastive.MatryoshkaDimensions)), len(contrastive.MatryoshkaDimensions))),
			matryoshka_dimensions_count: C.size_t(len(contrastive.MatryoshkaDimensions)),
			temperature_groups:          C.size_t(contrastive.TemperatureGroups),
			inverse_temperature:         C.float(contrastive.InverseTemperature),
			bias:                        C.float(contrastive.Bias),
			negatives:                   C.uform_negatives_t(contrastive.Negatives),
			ambient_repulsion_weight:    C.float(contrastive.AmbientRepulsionWeight),
			max_edges_per_device:        C.size_t(contrastive.MaxEdgesPerDevice),
			pair_loss:                   C.uform_pair_loss_form_t(contrastive.PairLoss),
		},
	}
	C.uform_trainer_init_vram(arguments)
	return trainerOf(block(&call, nil))
}

// Close frees the trainer, waiting for every deferred call naming it. It is idempotent.
func (trainer *Trainer) Close() error {
	if trainer == nil || trainer.handle == nil {
		return nil
	}
	C.uform_trainer_free(trainer.handle)
	trainer.handle = nil
	return nil
}

// endregion: Trainer

// region: Receivers

// trainerReceiver is a trainer and the executor its verbs run on.
type trainerReceiver struct {
	receiver[C.uform_trainer_t]
}

// TrainerOn is the receiver whose verbs block until their call is final. Binding-only.
type TrainerOn struct {
	trainerReceiver
}

// TrainerDeferredOn is the receiver whose verbs submit their calls and answer a Deferred, offering
// every verb of TrainerOn with the same arguments. Binding-only.
type TrainerDeferredOn struct {
	trainerReceiver
}

// On is the receiver whose verbs block on executor, the one the trainer was made on. Binding-only.
func (on TrainerOn) On(executor *Executor) TrainerOn {
	on.executor = executor
	return on
}

// Deferred is the receiver whose verbs submit their calls to executor, each answering a Deferred
// from the same arguments. Binding-only.
func (on TrainerOn) Deferred(executor *Executor) TrainerDeferredOn {
	on.executor = executor
	return TrainerDeferredOn{on.trainerReceiver}
}

// Step applies one optimizer step over this process's items, tagged by modalities, and every
// peer's, answering the pooled embeddings the loss scored when the config names an arena, and Rows
// of zero Count otherwise; every process of the run calls it once per step.
//
// Calls uform_trainer_step.
func (on TrainerOn) Step(modalities []Modality, contents Tape, config *TrainerStepConfig) (Rows, error) {
	var call callWithResult[Rows]
	return block(&call, on.step(&call, C.uform_completion_blocking_k, modalities, contents, config))
}

// Step submits TrainerOn.Step.
func (on TrainerDeferredOn) Step(modalities []Modality, contents Tape, config *TrainerStepConfig) (*Deferred[Rows], error) {
	deferred := new(Deferred[Rows])
	return submitted(deferred, on.step(&deferred.call, C.uform_completion_deferred_k, modalities, contents, config))
}

// Save writes the weights and the step to a safetensors checkpoint at path, which ModelLoadVram
// serves and TrainerInitVram resumes; only the process of index zero writes.
//
// Calls uform_trainer_save.
func (on TrainerOn) Save(path string, config *TrainerSaveConfig) error {
	var call callWithResult[struct{}]
	_, err := block(&call, on.save(&call, C.uform_completion_blocking_k, path, config))
	return err
}

// Save submits TrainerOn.Save.
func (on TrainerDeferredOn) Save(path string, config *TrainerSaveConfig) (*Deferred[struct{}], error) {
	deferred := new(Deferred[struct{}])
	return submitted(deferred, on.save(&deferred.call, C.uform_completion_deferred_k, path, config))
}

// endregion: Receivers

// region: Calls

func (receiver trainerReceiver) step(call *callWithResult[Rows], completion C.uform_completion_t, modalities []Modality,
	contents Tape, config *TrainerStepConfig) error {
	options := configOf(config)
	trainer, executor, outputsArena, err := receiver.lowerReadableInto(options.Arena)
	if err != nil {
		return err
	}
	count := contents.Entries()
	// The call has one count, so only this side sees the per-item inputs disagree with it.
	if uint(len(modalities)) != count {
		return fail(StatusShapeMismatch, "every item needs one modality")
	}
	if (len(options.Bags) != 0 && uint(len(options.Bags)) != count) ||
		(len(options.TemperatureGroups) != 0 && uint(len(options.TemperatureGroups)) != count) ||
		(len(options.Sides) != 0 && uint(len(options.Sides)) != count) {
		return fail(StatusShapeMismatch, "pass no bags, groups or sides, or one per item")
	}
	if options.Choices.Entries() != 0 && (options.Choices.Entries() != count || options.ChoiceTargets.Entries() != count) {
		return fail(StatusShapeMismatch, "every item needs its choices and their targets")
	}
	relations := options.Relations
	for _, edges := range [...]Edges{relations.Attract, relations.Repel, relations.Ignore} {
		if err := edges.check(count); err != nil {
			return err
		}
	}
	arguments := open[C.uform_trainer_step_t](call)
	var embeddings *C.uform_rows_view_t
	if options.Arena != nil {
		embeddings, call.result.count, call.result.arena = &call.result.raw, count, call.stamp(options.Arena, outputsArena)
	}
	*arguments = C.uform_trainer_step_t{
		trainer:            trainer,
		executor:           executor,
		outcome:            call.outcome(),
		completion:         completion,
		count:              C.size_t(count),
		modalities:         (*C.uform_modality_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(modalities)), len(modalities))),
		contents:           contents.lower(&call.call),
		bags:               call.uint32s(options.Bags),
		temperature_groups: call.uint32s(options.TemperatureGroups),
		sides:              (*C.uform_late_interaction_side_t)(call.pinSlice(unsafe.Pointer(unsafe.SliceData(options.Sides)), len(options.Sides))),
		choices:            options.Choices.lower(&call.call),
		choice_targets:     options.ChoiceTargets.lower(&call.call),
		relations: C.uform_relations_view_t{
			attract: call.edges(relations.Attract),
			repel:   call.edges(relations.Repel),
			ignore:  call.edges(relations.Ignore),
		},
		embeddings:     embeddings,
		arena:          outputsArena,
		outputs_scalar: C.uform_scalar_t(options.OutputsScalar),
		stats:          (*C.uform_trainer_step_stats_t)(call.optional(unsafe.Pointer(options.Stats))),
	}
	C.uform_trainer_step(arguments)
	return nil
}

func (receiver trainerReceiver) save(call *callWithResult[struct{}], completion C.uform_completion_t, path string,
	config *TrainerSaveConfig) error {
	options := configOf(config)
	trainer, executor, err := receiver.lower()
	if err != nil {
		return err
	}
	arguments := open[C.uform_trainer_save_t](call)
	*arguments = C.uform_trainer_save_t{
		trainer:    trainer,
		executor:   executor,
		outcome:    call.outcome(),
		completion: completion,
		path:       call.text(path),
		progress:   call.progress(options.Progress, options.ProgressSteps),
	}
	C.uform_trainer_save(arguments)
	return nil
}

// endregion: Calls
