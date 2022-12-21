import numpy as np

from core.entity import Entity
from see.judges import x_distance_between, y_distance_between, xy_distance_between


class Game:
    def __init__(self, canvas_size, distance_threshold=50):
        self.entities = {}
        self.distance_threshold = distance_threshold
        self.canvas_size = canvas_size

    def add_entity(self, entity_type, entity_init_kwargs, time):
        # if entity has not yet come into scope (empty list)
        if len(entity_init_kwargs) == 0:
            return

        for each_entity_init_kwargs in entity_init_kwargs:
            # if entity type already registered
            if entity_type in self.entities.keys():
                self.entities[entity_type].append(
                    Entity(
                        current_time=time,
                        canvas_size=self.canvas_size,
                        **dict(each_entity_init_kwargs, type=entity_type),
                    )
                )
            # if new entity type seen
            else:
                self.entities[entity_type] = [
                    Entity(
                        current_time=time,
                        canvas_size=self.canvas_size,
                        **dict(each_entity_init_kwargs, type=entity_type),
                    )
                ]

    def update_entity(self, entity_type, entity_update_kwargs, time, is_protagonist=False):
        assert entity_type in self.entities.keys(), "Updating before creation"

        # if len(self.entities[entity_type]) == 1:
        #     self.entities[entity_type][0].update_position(**entity_update_kwargs)
        # else:
        #     raise NotImplementedError

        # TODO: Make more robust based on estimated position with velocity and time
        for each_entity_update_kwargs in entity_update_kwargs:
            position = np.mean(each_entity_update_kwargs["new_hull"], axis=0)
            try:
                diff_min = np.min(
                    [
                        np.linalg.norm(entity.position - position)
                        for entity in self.entities[entity_type]
                        if entity.in_scope
                    ]
                )
                idx = [np.linalg.norm(entity.position - position) for entity in self.entities[entity_type]].index(
                    diff_min
                )
                if np.linalg.norm(self.entities[entity_type][idx].position - position) > self.distance_threshold:
                    # make new
                    if is_protagonist:
                        self.add_entity(
                            entity_type,
                            [
                                {
                                    "hull": each_entity_update_kwargs["new_hull"],
                                    "is_protagonist": True,
                                    "shoot_cooldown": 0,
                                }
                            ],
                            time,
                        )
                    else:
                        self.add_entity(
                            entity_type, [{"hull": each_entity_update_kwargs["new_hull"], "attended_to": -1}], time
                        )
                else:
                    self.entities[entity_type][idx].update_position(current_time=time, **each_entity_update_kwargs)

            except ValueError as e:
                # nothing in scope, got empty list to compare with
                if str(e) == "zero-size array to reduction operation minimum which has no identity":
                    if is_protagonist:
                        self.add_entity(
                            entity_type,
                            [
                                {
                                    "hull": each_entity_update_kwargs["new_hull"],
                                    "is_protagonist": True,
                                    "shoot_cooldown": 0,
                                }
                            ],
                            time,
                        )
                    else:
                        self.add_entity(
                            entity_type, [{"hull": each_entity_update_kwargs["new_hull"], "attended_to": -1}], time
                        )
                else:
                    raise e

        # Remove entities that are no longer in scope (if it has not been updated in this time step)
        for entity in self.entities[entity_type]:
            if entity.in_scope and time > 5 + entity.last_updated_time and not entity.static:
                entity.in_scope = False
                # if entity that was being tracked has gone out of scope
                if getattr(entity, "attended_to", None) == 0:
                    self.protagonist.is_tracking = False

    def has_entity(self, list_of_types, operator="and"):
        if operator == "and":
            for entity_type in list_of_types:
                # entity in env
                if entity_type in self.entities.keys():
                    # if in scope
                    if any([entity.in_scope for entity in self.entities[entity_type]]):
                        continue
                # entity not in env
                else:
                    return False
            # continued everytime
            return True
        elif operator == "or":
            for entity_type in list_of_types:
                # entity in env
                if entity_type in self.entities.keys():
                    # if in scope
                    if any([entity.in_scope for entity in self.entities[entity_type]]):
                        return True
                # entity not in env
                else:
                    continue
                # continued everytime
            return False
        else:
            raise NotImplementedError(f"Please define logic for {operator} operator")

    def get_entities(self, list_of_types):
        entities = []
        for entity_type in list_of_types:
            # entity registered in env
            if entity_type in self.entities.keys():
                # if in scope
                for entity in self.entities[entity_type]:
                    if entity.in_scope:
                        entities.append(entity)
        if len(entities) == 0:
            return False
        return entities

    @property
    def protagonist(self):
        # It is expected that only 1 protagonist is there, first one to be found
        # is returned.
        # Make sure OD is very robust (NMS should do the job for overlaps, but
        # similar patterns threshold should be well adjusted and kept.
        for _, entities_of_type in self.entities.items():
            for entity in entities_of_type:
                if entity.is_protagonist and entity.in_scope:
                    return entity
        return None

    @property
    def ground_plane(self):
        # It is expected that only 1 ground plane is there, first one to be found
        # is returned.
        # Make sure OD is very robust (NMS should do the job for overlaps, but
        # similar patterns threshold should be well adjusted and kept.
        for entity in self.entities["ground_plane"]:
            if entity.in_scope:
                return entity
        return None

    def nearest_x_to_protagonist(self, list_of_types, side="both"):
        min_dist = float("inf")
        closest_entity = None
        for entity_type in list_of_types:
            if self.has_entity([entity_type]):
                for entity in self.entities[entity_type]:
                    if side == "both":
                        if entity.in_scope and x_distance_between(self.protagonist, entity) < min_dist:
                            min_dist = x_distance_between(self.protagonist, entity)
                            closest_entity = entity
                    elif side == "right":
                        if entity.in_scope and 0 < x_distance_between(self.protagonist, entity) < min_dist:
                            min_dist = x_distance_between(self.protagonist, entity)
                            closest_entity = entity
                    elif side == "abs":
                        if entity.in_scope and abs(x_distance_between(self.protagonist, entity)) < min_dist:
                            min_dist = abs(x_distance_between(self.protagonist, entity))
                            closest_entity = entity
                    else:
                        raise NotImplementedError
        return closest_entity, min_dist

    def farthest_x_to_protagonist(self, list_of_types, side="abs", check_not_attended_to=False, threshold=0):
        max_dist = -float("inf")
        farthest_entity = None
        for entity_type in list_of_types:
            if self.has_entity([entity_type]):
                for entity in self.entities[entity_type]:
                    if side == "abs":
                        if entity.in_scope and threshold < abs(x_distance_between(self.protagonist, entity)) > max_dist:
                            if check_not_attended_to:
                                if entity.attended_to == -1:
                                    max_dist = abs(x_distance_between(self.protagonist, entity))
                                    farthest_entity = entity
                            else:
                                max_dist = abs(x_distance_between(self.protagonist, entity))
                                farthest_entity = entity
                    else:
                        raise NotImplementedError
        return farthest_entity, max_dist

    def nearest_y_to_protagonist(self, list_of_types, method="normal", check_not_attended_to=False):
        min_dist = float("inf")
        closest_entity = None
        for entity_type in list_of_types:
            if self.has_entity([entity_type]):
                for entity in self.entities[entity_type]:
                    if method == "both":
                        if entity.in_scope and abs(y_distance_between(self.protagonist, entity)) < min_dist:
                            if check_not_attended_to:
                                if entity.attended_to == -1:
                                    min_dist = abs(y_distance_between(self.protagonist, entity))
                                    closest_entity = entity
                            else:
                                min_dist = abs(y_distance_between(self.protagonist, entity))
                                closest_entity = entity

                    elif method == "negatives":
                        # negative and magnitude is smallest
                        if (
                            entity.in_scope
                            and y_distance_between(self.protagonist, entity) < 0
                            and abs(y_distance_between(self.protagonist, entity)) < min_dist
                        ):
                            min_dist = abs(y_distance_between(self.protagonist, entity))
                            closest_entity = entity
                    elif method == "normal":
                        if entity.in_scope and y_distance_between(self.protagonist, entity) < min_dist:
                            min_dist = y_distance_between(self.protagonist, entity)
                            closest_entity = entity
                    else:
                        raise NotImplementedError
        return closest_entity, min_dist

    def nearest_y_thresh_x_to_protagonist(self, list_of_types, method="abs", check_not_attended_to=False, x_thresh=20):
        min_dist = float("inf")
        closest_entity = None
        for entity_type in list_of_types:
            if self.has_entity([entity_type]):
                for entity in self.entities[entity_type]:
                    if method == "abs":
                        if entity.in_scope and abs(y_distance_between(self.protagonist, entity)) < min_dist:
                            if check_not_attended_to:
                                if entity.attended_to == -1:
                                    if abs(x_distance_between(self.protagonist, entity)) > x_thresh:
                                        min_dist = abs(y_distance_between(self.protagonist, entity))
                                        closest_entity = entity
                            else:
                                if abs(x_distance_between(self.protagonist, entity)) > x_thresh:
                                    min_dist = abs(y_distance_between(self.protagonist, entity))
                                    closest_entity = entity

        return closest_entity, min_dist

    def nearest_y_neg_thresh_x_to_protagonist(
        self,
        list_of_types,
        method="abs",
        check_not_attended_to=False,
        x_thresh=20,
    ):
        min_dist = float("inf")
        closest_entity = None
        for entity_type in list_of_types:
            if self.has_entity([entity_type]):
                for entity in self.entities[entity_type]:
                    if method == "abs":
                        if entity.in_scope and abs(y_distance_between(self.protagonist, entity)) < min_dist:
                            if check_not_attended_to:
                                if entity.attended_to == -1:
                                    if abs(x_distance_between(self.protagonist, entity)) < x_thresh:
                                        min_dist = abs(y_distance_between(self.protagonist, entity))
                                        closest_entity = entity
                            else:
                                if abs(x_distance_between(self.protagonist, entity)) < x_thresh:
                                    min_dist = abs(y_distance_between(self.protagonist, entity))
                                    closest_entity = entity
                    elif method == "above":
                        if (
                            entity.in_scope
                            and abs(y_distance_between(self.protagonist, entity)) < min_dist
                            and y_distance_between(self.protagonist, entity) < 0
                        ):
                            if check_not_attended_to:
                                if entity.attended_to == -1:
                                    if abs(x_distance_between(self.protagonist, entity)) < x_thresh:
                                        min_dist = abs(y_distance_between(self.protagonist, entity))
                                        closest_entity = entity
                            else:
                                if abs(x_distance_between(self.protagonist, entity)) < x_thresh:
                                    min_dist = abs(y_distance_between(self.protagonist, entity))
                                    closest_entity = entity
                    elif method == "below":
                        if entity.in_scope and 0 < y_distance_between(self.protagonist, entity) < min_dist:
                            if check_not_attended_to:
                                if entity.attended_to == -1:
                                    if abs(x_distance_between(self.protagonist, entity)) < x_thresh:
                                        min_dist = y_distance_between(self.protagonist, entity)
                                        closest_entity = entity
                            else:
                                if abs(x_distance_between(self.protagonist, entity)) < x_thresh:
                                    min_dist = y_distance_between(self.protagonist, entity)
                                    closest_entity = entity
                    else:
                        raise NotImplementedError

        return closest_entity, min_dist

    def nearest_xy_to_protagonist(self, list_of_types):
        min_dist = float("inf")
        closest_entity = None
        for entity_type in list_of_types:
            if self.has_entity([entity_type]):
                for entity in self.entities[entity_type]:
                    if entity.in_scope and xy_distance_between(self.protagonist, entity) < min_dist:
                        min_dist = xy_distance_between(self.protagonist, entity)
                        closest_entity = entity
        return closest_entity, min_dist

    def entity_with_flag(self, list_of_types, flag, value=None):
        found_entities = []
        for entity_type in list_of_types:
            if self.has_entity([entity_type]):
                for entity in self.entities[entity_type]:
                    if entity.in_scope and hasattr(entity, flag):
                        if getattr(entity, flag) == value:
                            found_entities.append(entity)
        if len(found_entities) > 0:
            return found_entities
        else:
            return None
