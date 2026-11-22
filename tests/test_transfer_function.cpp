#include "core/TransferFunction.h"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(TransferFunction, InterpolatesBetweenControlPoints) {
    const vs::TransferFunction tf({{0.0f, {0, 0, 0, 0}}, {1.0f, {1, 0.5f, 0.25f, 1}}});
    const vs::Rgba mid = tf.sample(0.5f);
    EXPECT_FLOAT_EQ(mid.r, 0.5f);
    EXPECT_FLOAT_EQ(mid.g, 0.25f);
    EXPECT_FLOAT_EQ(mid.b, 0.125f);
    EXPECT_FLOAT_EQ(mid.a, 0.5f);
}

TEST(TransferFunction, ClampsOutsideControlPoints) {
    const vs::TransferFunction tf({{0.2f, {1, 0, 0, 0.1f}}, {0.8f, {0, 1, 0, 0.9f}}});
    EXPECT_FLOAT_EQ(tf.sample(0.0f).r, 1.0f);
    EXPECT_FLOAT_EQ(tf.sample(-5.0f).a, 0.1f);
    EXPECT_FLOAT_EQ(tf.sample(1.0f).g, 1.0f);
}

TEST(TransferFunction, SortsUnorderedControlPoints) {
    const vs::TransferFunction tf({{1.0f, {1, 1, 1, 1}}, {0.0f, {0, 0, 0, 0}}});
    EXPECT_FLOAT_EQ(tf.sample(0.25f).a, 0.25f);
}

TEST(TransferFunction, TableMatchesSamples) {
    const auto tf = vs::TransferFunction::preset(1);
    const auto& table = tf.table();
    EXPECT_FLOAT_EQ(table.front().a, tf.sample(0.0f).a);
    EXPECT_FLOAT_EQ(table.back().a, tf.sample(1.0f).a);
}

TEST(TransferFunction, AllPresetsConstruct) {
    const auto names = vs::TransferFunction::presetNames();
    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
        EXPECT_NO_THROW((void)vs::TransferFunction::preset(i));
    }
    EXPECT_THROW((void)vs::TransferFunction::preset(static_cast<int>(names.size())), std::out_of_range);
}

TEST(TransferFunction, RejectsEmptyControlPoints) {
    EXPECT_THROW(vs::TransferFunction({}), std::invalid_argument);
}
