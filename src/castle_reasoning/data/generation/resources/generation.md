# Reversible construction generation

## Construction rule

The generator checks every primitive part addition, including the parts of a
multi-part root or growth operation. It rejects the complete operation if an
addition collides, obstructs vertical insertion, or fails independent support.
It records every proposal's outcome without mutating a rejected target.

For each added part, the contact-force equilibrium model assigns unit weight
to that part and zero weight to the earlier parts. Earlier parts transmit
forces but cannot act as balancing counterweights. Nonnegative contact forces
and per-part XY moment balance require the load to propagate through the
supporting contact hull at every interface. Contact corners are inset 5%.

If each independent load has a feasible force vector, their positive weighted
sum supports the actual structure. Removing a construction suffix removes its
load vectors; earlier witnesses never depended on later supports. This
certifies every construction prefix without counterweights. It is stronger
than testing only the completed structure or each part's own center of mass.

## Saved evidence

Accepted targets contain construction and disassembly orders, the stability
rule and inset, geometry fingerprints, and operation traces. The completed
target also passes the ordinary volume-weighted equilibrium check. Artifact
metadata retains the generator identifier `independent-load-construction-v1`.

The waypoint planner revalidates a supplied construction order's permutation,
vertical access, and prefix equilibrium, then reverses it for disassembly.
Targets without a witness use the existing bounded removal-order search.

## Limits

The guarantee assumes quasi-static gravity and horizontal nonadhesive contacts.
It deliberately excludes counterweight-dependent structures, including some
physically stable ones. The contact inset is a static margin, not an
acceleration bound. Vertical insertion certifies object access; it does not
certify gripper/arm access, staging capacity, or dynamic stability. Robot
execution and demonstration replay remain separate checks.

## Implementation

The installed Python implementation is under
`castle_reasoning.data.generation`: `geometry.construction_rejection` checks
each addition with `geometry.equilibrium`; `generator.generate_pool` records
proposals and accepted targets; `waypoints.plan` checks or searches removal
orders. Evaluate benchmark results against the existing published task inputs.
Generating new candidates does not reproduce the historical campaign.
