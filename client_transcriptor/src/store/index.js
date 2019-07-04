import Vue from 'vue'
import Vuex from 'vuex'
import * as getters from './getters'
import * as actions from './actions'

Vue.use(Vuex)

const state = {
  models: [
    { 'id': 1, 'name': 'Deepspeech' }
  ]
}

export default new Vuex.Store({
  state: state,
  getters: getters,
  actions: actions,
  mutations: {
    addModel (state, model) {
      state.models.push(model)
    }
  }
})
