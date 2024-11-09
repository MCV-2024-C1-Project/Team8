import numpy as np

def retrieve_similar_images(sift, flann, cropped_query_image_list_d1, gt_list, museum_descriptors, K=1):
    results = []
    print(f"\nRetrieving similar images using SIFT descriptor:")
    for idx, (query_images, gt_tuple) in enumerate(zip(cropped_query_image_list_d1, gt_list)):
        print(f"Query {idx}")
        query_result = []
        for img_idx, image in enumerate(query_images):
            print(f" - Image {img_idx}")
            query_descriptor = sift.compute(np.array(image))
            img_results = get_img_results(query_descriptor, museum_descriptors, flann)

            best_candidate = determine_best_candidate(img_results, query_descriptor, museum_descriptors, flann, K)

            if best_candidate[0][1] < 0.1:
                best_candidate = [(-1, 1)]
            query_result.append(best_candidate)

            # Print and compare with ground truth
            print(f" Results : {query_result[img_idx][0]}")
            print(f" GT : {gt_tuple[img_idx]}")
            if query_result[img_idx][0][0] != gt_tuple[img_idx]:
                print("Mismatch detected")
            print("############\n\n")
        results.append(query_result)

    return results


def compute_similarity(descriptor1, descriptor2, flann):
    matches = flann.knnMatch(descriptor1[1], descriptor2[1], k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    similarity_score = len(good) / len(matches) if matches else 0
    return similarity_score


def get_img_results(query_descriptor, museum_descriptors, flann, reverse=False):
    img_results = []
    for museum_idx, db_descriptor in enumerate(museum_descriptors):
        if db_descriptor[1] is None:
            continue

        if reverse:
            similarity_score = compute_similarity(db_descriptor, query_descriptor, flann)
        else:
            similarity_score = compute_similarity(query_descriptor, db_descriptor, flann)
        img_results.append((museum_idx, similarity_score))
    img_results.sort(key=lambda x: x[1], reverse=True)
    return img_results



def determine_best_candidate(img_results, query_descriptor, museum_descriptors, flann, K):
    if len(img_results) > 1:
        top_score = img_results[0][1]
        second_score = img_results[1][1]
        relative_gap = (top_score - second_score) / top_score if top_score > 0 else 0
        print("   - Relative gap: ", relative_gap)
        ambiguous = relative_gap < 0.2
    else:
        ambiguous = False

    if ambiguous:
        print("Ambiguous result detected")
        reverse_results = get_img_results(query_descriptor, museum_descriptors, flann, True)
        reverse_top_score = reverse_results[0][1]
        if (reverse_top_score - second_score) / reverse_top_score >= 0.2:
            if reverse_top_score > top_score:
                best_candidate = reverse_results[0:K]
            else:
                best_candidate = img_results[0:K]
        else:
            print("Detected as not found")
            best_candidate = [(-1, 1)]  # Mark as ambiguous
    else:
        best_candidate = img_results[0:K] if img_results else [(-1, 1)]
    return best_candidate

