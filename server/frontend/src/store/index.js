import Vue from 'vue'
import Vuex from 'vuex'
import * as getters from './getters'
import * as actions from './actions'

Vue.use(Vuex)

const state = {
  models: [
  ]
}

export default new Vuex.Store({
  state: state,
  getters: {
    getPickedModels: state => {
      return state.models
    }
  },
  actions: actions,
  mutations: {

    addModel (state, model) {
      state.models.push(model)
    },

    removeModel (state, model_id) {
      var i = 0

      for (; i < state.models.length; i++) {
        if (this.state.models[i].id === model_id) {
          break
        }
      }

      if (i !== state.models.length) {
        state.models.splice(i, 1)
      }
    }

  }
})
