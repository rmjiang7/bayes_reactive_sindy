import numpy as np
import itertools


def encode(stoichiometry, rate):
    return ",".join(
        stoichiometry.astype(int).astype(str)
    ) + "-" + ",".join(
        rate.astype(int).astype(str)
    )


def generate_valid_reaction_basis(species):

    species_to_idx_map = {s: i for i, s in enumerate(species)}
    stoichiometry = []
    rates = []
    reaction_description = []
    # Mono-molecular
    for i, s in enumerate(species):
        creation = np.zeros(len(species) + 1)
        creation[i] = 1
        creation_rate = np.zeros(len(species) + 1)
        creation_rate[-1] = 1
        destruction = np.zeros(len(species) + 1)
        destruction[i] = -1
        stoichiometry += [creation, destruction]
        destruction_rate = np.zeros(len(species) + 1)
        destruction_rate[i] = 1
        rates += [creation_rate, destruction_rate]
        reaction_description += ["0 -> {}".format(s),
                                 "{} -> 0".format(s)]

        for j, z in enumerate(species):
            if z != s:
                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += -1
                reaction[species_to_idx_map[z]] += 1
                desc = "{} -> {}".format(s, z)
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[s]] += 1
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(desc)

                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += 0
                reaction[species_to_idx_map[z]] += 1
                desc = "{} -> {} + {}".format(s, s, z)
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[s]] += 1
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(desc)

                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += -2
                reaction[species_to_idx_map[z]] += 1
                desc = "2{} -> {}".format(s, z)
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[s]] += 2
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(desc)

                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += 2
                reaction[species_to_idx_map[z]] -= 1
                desc = "{} -> 2{}".format(z, s)
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[z]] += 1
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(desc)

            else:
                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += 1
                desc = "{} -> 2{}".format(s, s)
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[s]] += 1
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(desc)

        for z in itertools.combinations(species, 2):
            if s != z[0] and s != z[1]:
                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += -1
                reaction[species_to_idx_map[z[0]]] += 1
                reaction[species_to_idx_map[z[1]]] += 1
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[s]] += 1
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(
                    "{} -> {} + {}".format(s, z[0], z[1]))

    # Bi-molecular
    for i, s in enumerate(itertools.combinations(species, 2)):
        for j, z in enumerate(species):

            reaction = np.zeros(len(species) + 1)
            reaction[species_to_idx_map[s[0]]] += -1
            reaction[species_to_idx_map[s[1]]] += -1
            if z == s[0]:
                reaction[species_to_idx_map[s[0]]] += 2
                desc = "{} + {} -> 2{}".format(s[0], s[1], s[0])
            elif z == s[1]:
                reaction[species_to_idx_map[s[1]]] += 2
                desc = "{} + {} -> 2{}".format(s[0], s[1], s[1])
            else:
                reaction[species_to_idx_map[z]] += 1
                desc = "{} + {} -> {}".format(s[0], s[1], z)
            reaction_rate = np.zeros(len(species) + 1)
            reaction_rate[species_to_idx_map[s[0]]] += 1
            reaction_rate[species_to_idx_map[s[1]]] += 1
            stoichiometry.append(reaction)
            rates.append(reaction_rate)
            reaction_description.append(desc)

        for z in itertools.combinations(species, 2):
            reaction = np.zeros(len(species) + 1)
            reaction[species_to_idx_map[s[0]]] += -1
            reaction[species_to_idx_map[s[1]]] += -1
            reaction[species_to_idx_map[z[0]]] += 1
            reaction[species_to_idx_map[z[1]]] += 1
            reaction_rate = np.zeros(len(species) + 1)
            reaction_rate[species_to_idx_map[s[0]]] += 1
            reaction_rate[species_to_idx_map[s[1]]] += 1
            if not (reaction == 0).all():
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(
                    "{} + {} -> {} + {}".format(s[0], s[1], z[0], z[1]))

    reaction_mappings = {}
    for i in range(len(stoichiometry)):
        reaction_mappings[encode(stoichiometry[i], rates[i])] = i
    return np.vstack(stoichiometry), np.vstack(rates), reaction_description, reaction_mappings
